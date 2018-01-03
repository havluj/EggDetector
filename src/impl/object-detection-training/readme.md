we have 1890 eggdetections (1701 train + 189 test)

setup the environment:
clone https://github.com/tensorflow/models
`cd models/research`
`python3.5 setup.py`
Add these lines at the end of your ~/.bashrc file: 
```
export MODELS= path_to_models_directory (in this case it's ~/Desktop/models)
export PYTHONPATH=$MODELS:$MODELS/slim
export OBJ_DET=$MODELS/object_detection
```
`sudo pip3.5 install tensorflow-gpu`

correct image names in xml detections
```
for fullfile in *.jpg; do
	filename=$(basename "$fullfile")
	filename="${filename%.*}"
	echo "$filename".xml
	awk -v var="$filename" 'NR==3{$0="\t<filename>"var".jpg</filename>"}1;' "$filename".xml > temp.xml && mv temp.xml "$filename".xml
done

for fullfile in *.png; do
	filename=$(basename "$fullfile")
	filename="${filename%.*}"
	echo "$filename".xml
	awk -v var="$filename" 'NR==3{$0="\t<filename>"var".png</filename>"}1;' "$filename".xml > temp.xml && mv temp.xml "$filename".xml
done
```

generate csv files from xml
`python3.5 xml_to_csv.py`

generate train data
`python3.5 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record`

generate test data
`python3.5 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record`

train the network
`$OBJ_DET/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config`

export inference graph
```
python3.5 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-7918 \
    --output_directory frozen_graph
```

train on cloud
```
gcloud ml-engine jobs submit training object_detection_${version_unique_ID} \
    --job-dir=gs://eggdata/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --config object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir=gs://eggdata/train \
    --pipeline_config_path=gs://eggdata/data/faster_rcnn_resnet101_coco.config
```