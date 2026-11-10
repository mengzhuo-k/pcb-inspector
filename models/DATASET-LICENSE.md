# License & Attribution

## SMD DefDB (whole-board defect classifier)

Moshayedi, Ata Jahangir; Khan, Amir Sohail (2026). *SMD DefDB: (PCB SMD componet)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20001570

Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The dataset contains 119 source photos of SMD assemblies, expanded to 1190 images across seven classes (damaged_component, excess_solder, good_solder, misaligned_component, missing_component, poor_solder, solder_bridge). It provides image-level labels only; no bounding boxes. This project derives the weights `models/smd_defect_cls_v1.pt` from it and keeps the attribution above in the README and model card.

## Merged solder-joint detection dataset

This dataset merges several independent real-world PCB assembly/solder-joint photo sources, manually reviewed, deduplicated, and re-annotated into a consistent 9-class YOLO schema by the dataset author (ParsaVictor).

Released for **research, educational, and internal engineering use**. For commercial use or redistribution of the complete merged package, please contact the author via their Hugging Face profile.

No warranty is provided on label accuracy; labels were manually reviewed but may contain errors — please report issues via the repository's community/discussion tab.
