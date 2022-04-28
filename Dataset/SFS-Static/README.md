# SFS-Static Dataset
__________________________________________________
We introduce our SFS-Static dataset which is collected from 46 classrooms from 12 buildings on The University of Michigan’s campus, resulting of approximately 200 minutes audio. 


### ProcessedData
Folder `ProcessedData`: contains 46 classrooms. For each classroom, it has 16 − 30 10s audio clips which are recorded in the different positions. For each audio, it has directory structure:

- `RGB.png`: RGB frames in `.png` format, with size of `640 x 480`
- `Depth.png`: Depth frames in `.png` format with `uint16`, with size of `640 x 480`. To cover the depth map in meter unit: `depth * meta_data['Max depth in map'] / (2**16)`
- `sound.wav`: aligned soundtrack of sample rate 16K for corresponding video
- `annotation.json`: meta data and labels