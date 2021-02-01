# Repo for the python code that runs generates length estimates for oceanruler.org

Sample run:
python poseidon.py --ref_object square --ref_object_units cm --ref_object_size 10.0 --fishery_type finfish --uuid  "12345678Test" --username testRun --email "None given" --original_filename "fuu.bar" --original_size 0 --loc_code "none" --measurement_direction length --image sample_photos/finfish/288_KLPBS_315.jpg --show True
--show True will pop up the results so you can see how the model(s) are operating
If you see this: ModuleNotFoundError: No module named 'cv2', you need to run conda activate fastai-cpu
