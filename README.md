# abalone_length
Repo for the python code that runs generates length estimates for oceanruler.org

Sample run:
python poseidon.py --ref_object square --ref_object_units cm --ref_object_size 10.0 --fishery_type finfish --uuid  "12345678Test" --username testRun --email "None given" --original_filename "fuu.bar" --original_size 0 --loc_code "none" --measurement_direction length --image sample_photos/finfish/288_KLPBS_315.jpg --show True
--show True will pop up the results so you can see how the model(s) are operating
If you see this: ModuleNotFoundError: No module named 'cv2', you need to run conda activate fastai-cpu

# Env variable $ML_PATH needs to be set for the python scripts to find the machine learning libraries:
ML_PATH=/home/ubuntu/abalone_length/machine_learning
Note: this dir also contains the fastai libraries which are currently at 0.7 -- this is, unfortunately, old now but updating will break the current model running code
ml_data -> contains all of the model data
masks/ contains all of the temporary masking data
Note: this accumulates over time, need to set up a chron job to clear it out periodically
tmp/ contains the images that are downloaded from the web client.
Note: also accumulates over time, need to set up a chron job to clear it
The code to actually generate the models based on input data can be found in this directory. To generate new models, run and modify the ablob_load_model.py or ablob_load_model.ipynb
To learn more about running the model, visit here: https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652
To run the models, you need a gpu. I’ve used the on-demand gpus available at paperspace to generate them
/home/ubuntu/abalone_length/machine_learning needs to be on the $PYTHONPATH env variable. It should already be set
Each run executes executes in roughly this order:
ocean-ruler-server/index.js -> poseidon.py -> lambda_function.py -> contour_utils.py, drawing.py
Poseidon.py loads the existing machine learning models, lambda_function loads and clips everything, contour_utils does the computer vision/edge detection work, and drawing computes the size based on the results
