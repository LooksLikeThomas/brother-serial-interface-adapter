```
              /|
             /+|
 _________  /++|_____    _____  _    _ ____  _         __
|++++++++/||###|:::::|  |  __ `| |  | |  _ `| |  __   / /
|+++++++/#||###|:::::|  | |  | | |__| | |_) | | /  | / /
|++++++/##||##/::::::|  | |  | |  __  |  _ -| |/ | |/ /
|+++++|###||#/:::::::|  | |__| | |  | | |_) | | /| | /
|=====|###||/========|  |_____/|_|  |_|____/|__/ |__/
      |::/
      |:/      Brother Serial Interface Adapter
      |/
```


A modern USB adapter for vintage Brother typewriters, reverse engineering the proprietary "Brother Serial Interface System" from the 1980s.

---
# Brother Serial Interface Adapter Project

In the 1980s, electronic typewriters represented a alternative to early PC systems. Brother developed the IF-60, a hardware interface that converted compatible typewriters into daisy wheel printers or TTY terminals by regulating, buffering, and adapting communication between the proprietary "Brother Serial Interface System" and the RS232 or Centronics interfaces of PCs.

This project systematically reverse engineers the Brother Serial Interface System through hardware-level analysis using a logic analyzer, then develops a USB adapter prototype to serve as a functional replacement for the IF-60 box. The adapter will connect compatible typewriters to current hardware via a standardized USB-CDC interface (virtual COM port).

---

## Project Structure

```
brother-serial-interface-adapter/
├── protocol-analysis/    # Phase 1: Signal analysis and protocol reverse engineering
├── firmware/             # Phase 2: Microcontroller firmware for USB adapter
├── hardware/             # Physical specifications, CAD models, and schematics
└── docs/                 # Reference materials, manuals, and patents
```

**protocol-analysis/**  
Contains captured signal data and Jupyter notebooks for signal capture and analysis, along with the `analysis_lib` Python package providing decoders and visualization tools for the Brother protocol.  

**firmware/**  
PlatformIO-based microcontroller firmware that translates between the proprietary Brother protocol and USB serial communication. This will run on the Arduino hardware.

**hardware/**  
3D models for connectors, pinout documentation, and circuit schematics for the physical adapter interface.

**docs/**  
User manuals, service manuals, and patents organized by device model providing reference material for the reverse engineering process.

---

## About

**Status**  
This project is under active development. The protocol analysis phase is in progress.

**Author**  
This project is part of a student project at the Baden-Württemberg Cooperative State University ([DHBW Mannheim](https://www.dhbw-mannheim.de/)), conducted by [Thomas Henseler](https://www.linkedin.com/in/thomas-henseler-a101172b8/) and supervised by [Prof. Dr.-Ing. Johannes Bauer](https://www.mannheim.dhbw.de/profile/bauer).

## License
The firmware, hardware designs, and analysis tools in this project are open source under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

**Third-Party Copyrights**  
Original Brother Service Manuals and documentation referenced in this project remain the copyright of Brother Industries, Ltd. They are not covered by the MIT license of this project.

**Trademark Acknowledgement**  
Brother, Brother Serial Interface System, IF-60, IF-50, and all related product names are trademarks of Brother Industries, Ltd. This project is an independent educational endeavor and is not affiliated with Brother Industries, Ltd.
