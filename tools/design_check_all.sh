#!/usr/bin/env bash
# Re-run tools/design_check.py on every WACV image that has a design file (designfiles/, git-ignored; sources in memory/ notes).
# HuMANDATA .cen files come from tools/pdf_to_centroid.py (run with an env that has PyMuPDF); MicroZed ones from its --altium
# mode, two rounds (all parts -> check both photos -> drop the parts the other photo confirmed -> check again), see below.
cd "${CROSSCLASS:-$HOME/projects/crossclass}" || exit 1
PY=${PY:-$HOME/venvs/gdino_fewshot/bin/python}
while read -r img src side; do
  [ -z "$img" ] && continue
  $PY -I tools/design_check.py --image "$img" --centroid "designfiles/$src" --side "$side" | sed -n 2p | sed "s/^/$img: /"
done <<LIST
ATTIOT_Bottom FRDM-K64F/raw/CEN-28163_E.smt top
ArduinoMega_Top ArduinoMega2560/raw/Arduino_MEGA_2560-Rev3.brd top
DuetWIFI_Top Smoothieboard/Smoothieboard-master/smoothieboard-5driver.brd top
ML450_Top1 ML450/raw/IPC-D-356.txt top
ML450_Top2 ML450/raw/IPC-D-356.txt top
ML450_Bottom1 ML450/raw/IPC-D-356.txt bottom
ML450_Bottom2 ML450/raw/IPC-D-356.txt bottom
XCM-307A_Top HUMANDATA/XCM307-PCB-B-B.cen top
XCM-307A_Bottom HUMANDATA/XCM307-PCB-B-B.cen bottom
ACM-109_Top HUMANDATA/ACM109-PCB-B-B.cen top
ACM-109_Bottom HUMANDATA/ACM109-PCB-B-B.cen bottom
EDA-008_Top HUMANDATA/EDA008-PCB-C-C.cen top
EDA-008_Bottom HUMANDATA/EDA008-PCB-C-C.cen bottom
MicroZed_Top MicroZed/RevB_top.cen top
MicroZed_Bottom MicroZed/RevB_bottom.cen bottom
LIST
