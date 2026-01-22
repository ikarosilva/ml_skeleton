"""SQLite-based storage for song embeddings with multi-version support.

This module provides persistent storage for audio embeddings, allowing:
- Multiple encoder versions per song (A/B testing)
- Efficient batch operations
- Fast retrieval by filename and version
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional
import io
import time


class EmbeddingStore:
    """SQLite-backed storage for song embeddings.
    
    Features:
    - Multi-version support: Store embeddings from different encoder versions
    - Batch operations: Efficient insert/update for large datasets
    - Thread-safe: Uses connection per operation
    
    Schema:
        CREATE TABLE embeddings (
            filename TEXT NOT NULL,
            model_version TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (filename, model_version)
        )
    """
    
    def __init__(self, db_path: str = "./embeddings.db"):
        """Initialize embedding store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create table with multi-version support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                filename TEXT NOT NULL,
                model_version TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (filename, model_version)
            )
        """)
        
        # Create indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_version 
            ON embeddings(model_version)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated_at 
            ON embeddings(updated_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes."""
        buffer = io.BytesIO()
        np.save(buffer, embedding, allow_pickle=False)
        return buffer.getvalue()
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=False)
    
    def store_embedding(
        self,
        filename: str,
        embedding: np.ndarray,
        model_version: str = "default"
    ):
        """Store single embedding.
        
        Args:
            filename: Song filename (from Clementine DB)
            embedding: Embedding vector (1D numpy array)
            model_version: Encoder model version identifier
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        embedding_blob = self._serialize_embedding(embedding)
        embedding_dim = len(embedding)
        current_time = time.time()
        
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings 
            (filename, model_version, embedding, embedding_dim, created_at, updated_at)
            VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT created_at FROM embeddings 
                              WHERE filename = ? AND model_version = ?), ?),
                    ?)
        """, (
            filename, model_version, embedding_blob, embedding_dim,
            filename, model_version, current_time,  # For COALESCE
            current_time  # updated_at
        ))
        
        conn.commit()
        conn.close()
    
    def store_embeddings_batch(
        self,
        data: list[tuple[str, np.ndarray]],
        model_version: str = "default",
        batch_size: int = 1000
    ):
        """Batch insert/update embeddings (much faster than individual inserts).
        
        Args:
            data: List of (filename, embedding) tuples
            model_version: Encoder model version identifier
            batch_size: Number of embeddings per transaction
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        current_time = time.time()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Prepare batch data
            batch_data = []
            for filename, embedding in batch:
                embedding_blob = self._serialize_embedding(embedding)
                embedding_dim = len(embedding)
                batch_data.append((
                    filename, model_version, embedding_blob, embedding_dim,
                    filename, model_version, current_time, current_time
                ))
            
            # Execute batch insert
            cursor.executemany("""
                INSERT OR REPLACE INTO embeddings 
                (filename, model_version, embedding, embedding_dim, created_at, updated_at)
                VALUES (?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM embeddings 
                                  WHERE filename = ? AND model_version = ?), ?),
                        ?)
            """, batch_data)
            
            conn.commit()
        
        conn.close()
    
    def get_embedding(
        self,
        filename: str,
        model_version: str = "default"
    ) -> Optional[np.ndarray]:
        """Retrieve embedding by filename and model version.
        
        Args:
            filename: Song filename
            model_version: Encoder model version
            
        Returns:
            Embedding vector or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding FROM embeddings
            WHERE filename = ? AND model_version = ?
        """, (filename, model_version))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return self._deserialize_embedding(row[0])
    
    def get_embeddings_batch(
        self,
        filenames: list[str],
        model_version: str = "default"
    ) -> dict[str, np.ndarray]:
        """Retrieve multiple embeddings efficiently.
        
        Args:
            filenames: List of song filenames
            model_version: Encoder model version
            
        Returns:
            Dictionary mapping filename -> embedding
            Only includes filenames that have embeddings
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create placeholders for SQL IN clause
        placeholders = ','.join('?' * len(filenames))
        
        cursor.execute(f"""
            SELECT filename, embedding FROM embeddings
            WHERE model_version = ? AND filename IN ({placeholders})
        """, [model_version] + filenames)
        
        results = {}
        for filename, embedding_blob in cursor.fetchall():
            results[filename] = self._deserialize_embedding(embedding_blob)
        
        conn.close()
        return results
    
    def get_all_versions(self, filename: str) -> list[str]:
        """Get all model versions that have embeddings for this file.
        
        Args:
            filename: Song filename
            
        Returns:
            List of model version identifiers
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_version FROM embeddings
            WHERE filename = ?
            ORDER BY updated_at DESC
        """, (filename,))
        
        versions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return versions
    
    def has_embedding(
        self,
        filename: str,
        model_version: str = "default"
    ) -> bool:
        """Check if embedding exists.
        
        Args:
            filename: Song filename
            model_version: Encoder model version
            
        Returns:
            True if embedding exists
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 1 FROM embeddings
            WHERE filename = ? AND model_version = ?
            LIMIT 1
        """, (filename, model_version))
        
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def count(self, model_version: Optional[str] = None) -> int:
        """Count total number of embeddings.
        
        Args:
            model_version: If provided, count only for this version
            
        Returns:
            Number of embeddings
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if model_version is None:
            cursor.execute("SELECT COUNT(*) FROM embeddings")
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_version = ?",
                (model_version,)
            )
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear(self, model_version: Optional[str] = None):
        """Delete embeddings.
        
        Args:
            model_version: If provided, delete only this version.
                          If None, delete all embeddings.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if model_version is None:
            cursor.execute("DELETE FROM embeddings")
        else:
            cursor.execute(
                "DELETE FROM embeddings WHERE model_version = ?",
                (model_version,)
            )
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> dict:
        """Get storage statistics.
        
        Returns:
            Dictionary with statistics:
            - total_embeddings: Total number of embeddings
            - unique_songs: Number of unique songs
            - model_versions: List of all model versions
            - db_size_mb: Database file size in MB
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total embeddings
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]
        
        # Unique songs
        cursor.execute("SELECT COUNT(DISTINCT filename) FROM embeddings")
        unique_songs = cursor.fetchone()[0]
        
        # Model versions
        cursor.execute("SELECT DISTINCT model_version FROM embeddings ORDER BY model_version")
        model_versions = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Database file size
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        
        return {
            "total_embeddings": total_embeddings,
            "unique_songs": unique_songs,
            "model_versions": model_versions,
            "db_size_mb": round(db_size_mb, 2)
        }
