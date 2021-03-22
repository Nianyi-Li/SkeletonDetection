export PROJECT_HOME='/home/scott/Documents/sidehustle/covid/SMAP/'
python3 test.py -p "/home/scott/Documents/sidehustle/covid/SMAP_WEIGHTS/SMAP_model.pth" \
-t run_inference \
-d test \
-rp "/home/scott/Documents/sidehustle/covid/SMAP_WEIGHTS/RefineNet.pth" \
--batch_size 16 \
--do_flip 1 \
--dataset_path "/home/scott/Documents/sidehustle/covid/data/aquarium"
