#!/bin/bash
# check_power_status.sh
# Checks for UPS connectivity and battery status using common Linux tools.

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check for Container environment
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}⚠ WARNING: You appear to be inside a Docker container.${NC}"
    echo "UPS monitoring should typically be configured on the HOST system"
    echo "so it can shut down the physical machine."
    echo "USB devices may not be visible here unless passed through."
    echo ""
fi

echo -e "${YELLOW}=== 1. Checking USB Devices ===${NC}"
# Check if lsusb is installed
if ! command -v lsusb &> /dev/null; then
    echo -e "${RED}✗ 'lsusb' command not found.${NC}"
    echo "  Install it to check USB devices:"
    echo "  Debian/Ubuntu: apt-get update && apt-get install -y usbutils"
# Look for common UPS keywords in USB device list
elif lsusb | grep -i -E "ups|power|battery|apc|cyberpower|eaton|tripp"; then
    echo -e "${GREEN}✓ UPS device detected on USB bus${NC}"
    lsusb | grep -i -E "ups|power|battery|apc|cyberpower|eaton|tripp"
else
    echo -e "${RED}✗ No obvious UPS device found on USB. Check cable connection.${NC}"
fi
echo ""

echo -e "${YELLOW}=== 2. Checking Power Supply Class (/sys/class/power_supply) ===${NC}"
if [ -d "/sys/class/power_supply" ] && [ "$(ls -A /sys/class/power_supply)" ]; then
    for supply in /sys/class/power_supply/*; do
        echo "Device: $(basename "$supply")"
        cat "$supply/uevent" | grep -E "STATUS|CAPACITY|MODEL|MANUFACTURER|ONLINE" | sed 's/^/  /'
    done
else
    echo "No power supply devices exposed in /sys/class/power_supply"
fi
echo ""

echo -e "${YELLOW}=== 3. Checking 'upower' (Standard Linux Power Manager) ===${NC}"
if command -v upower &> /dev/null; then
    # List devices and show status
    upower -e | while read -r device; do
        # Filter for UPS or Battery
        if upower -i "$device" | grep -q -E "ups|battery"; then
            echo "Device: $device"
            upower -i "$device" | grep -E "model|state|percentage|time to empty|online" | sed 's/^/  /'
        fi
    done
else
    echo "upower command not found (install 'upower' to see system power status)."
fi
echo ""

echo -e "${YELLOW}=== 4. Checking 'apcaccess' (APC UPS Daemon) ===${NC}"
if command -v apcaccess &> /dev/null; then
    apcaccess status | grep -E "STATUS|LINEV|LOADPCT|BCHARGE|TIMELEFT"
else
    echo "apcaccess not found (install 'apcupsd' if using an APC UPS)"
fi
echo ""

echo -e "${YELLOW}=== Recommendation ===${NC}"
echo "If your UPS is detected in step 1 but not step 3 or 4, you likely need to install"
echo "monitoring software like 'apcupsd' (for APC) or 'nut' (Network UPS Tools)."