This project showcases snippets from a prototype 3D graphics engine (based on the https://github.com/ecann/RenderPy) capable of rendering a
rotating VR headset model driven by a dataset of IMU readings. The core
motion tracking system currently leverages dead reckoning to calculate the
orientation based on the gyroscope data. and complementary filters for tilt
correction (using accelerometer data) and mitigating yaw drift (using
magnetometer data) have been implemented to enhance accuracy.
