# act_tubelet_dataset_gen
generate the action centric tubelet dataset


#### BBOX Variations
1. org
2. union
3. uniform

#### Processing Variations
1. parllel
1. sequential

#### DIR format
{SOURCE_DATASET}-{ORG_VIDEO_NAME}-{ACTIVITY_NAME}_act{ACT_NO}_p{PART_NO}

#### Known Issues
1. When using *org* bbox variation, few frames may skipped due to errors. 
    1.1 Need to write script to check for these and make duplicates for missing frames if necessary 
2. OKUTAMA tubelets
    There is a issue with Union bbox variation, for this dataset, we have to stick with "org" bbox variation
    This is mainly due to large camera variations, tubelet becoming too wide.
    One more thing is, in okutama **camera is moving** i.e its on drone, so taking union of bounding boxes don't generate the expected results
3. UCF ARG 
    Torchvision models are unable to detect person in the rooftop and aerial videos (or sometimes detecting vehicle as pedestrians)