# Toxic Comment Classification
Implementation of the BERT (Devlin et al., 2019) architecture, inlcuding the attention-mechanism, to classify toxic comments. The data comes from the 2017 Kaggle toxic comment classification challenge (Jeffrey et al., 2017). It is a classification task, requiring to assign multiple labels for seven types of toxicity (including ”non-toxic”) to the comments.BERT was fine-tuned in the toxic comment classification downstream task; pretrained weights where used instead of a regular pretraining. 

## Run the program

### With apptainer
1.  Image pml.sif to be obtained from: https://drive.google.com/file/d/14OY0eLTDKsIWZLcCUGUsiWxxkv_ljHUz/view?usp=sharing (exceeds GitHub maximum size)

2.   To run program in apptainer, use: 
    ```bash
    apptainer run --nv -B /home/space/datasets/toxic_comment:/home/space/datasets/toxic_comment pml.sif python main.py
    ```
3.  Output files are saved in `./output_folder` 

### Without apptainer (local)
1.  Create virtual environment and actiavte it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure dataset path `TOXIC` is configures in `params.py`.
4.  Run:
    ```bash
    python main.py
    ```