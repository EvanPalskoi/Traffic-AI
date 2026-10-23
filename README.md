# Traffic AI

Vehicle detection, multi-camera tracking visualization, and reinforcement learning traffic-signal control.

This project includes two demos:

- **Video analysis:** Detects and tracks vehicles within four camera views using RF-DETR, with adjustable camera synchronization, vehicle counts, tracking logs, and demand-based traffic-signal visualization. An optional YOLO model is also provided.
- **SUMO simulation:** Uses a trained Proximal Policy Optimization (PPO) agent to select traffic-signal phases in a simulated intersection.

Built with Python, ONNX Runtime, OpenCV, Ultralytics, SUMO-RL, and Stable-Baselines3.
