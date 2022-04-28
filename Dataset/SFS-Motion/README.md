# SFS-Motion Dataset
__________________________________________________
We introduce our SFS-Montion dataset which contain 222 videos recorded from several buildings in UM North Campus. 

### RawData 
Folder `RawData`: contains 222 folders. For each scene folder, it has directory structure:
- `video.bag`: rosbag file contain depth frame and RGB frames with FPS=15
- `sound.wav`: unaligned soundtrack of sample rate 96K for corresponding video 
- `td.npy`: time difference of audio stream and vision stream

We don't public the raw data. However, you can send a request by e-mail to `czyang@umich.edu` and state your purpose clear. 

### ProcessedData
Folder `ProcessedData`: contains 222 scenes (video FPS=15). For each scene folder, it has directory structure:

- `RGB`: RGB frames in `.png` format, with size of `640 x 480`
- `Depth`: Depth frames in `.png` format with `uint16`, with size of `640 x 480`. To cover the depth map in meter unit: `depth * 10 / (2**16)`
- `Audio`: aligned soundtrack of sample rate 16K for corresponding video
