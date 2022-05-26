# p6-projekt
## Grasp Affordance Estimator
![alt text](https://i.imgur.com/khUi3fr.png)

### How to run. 

- Download dataset from [3D-AffordanceNet](https://andlollipopde.github.io/3D-AffordanceNet/#/) and add it do the directory 
```'C:\\data_for_learning'```
- Run ```mltraining.py``` to extract the features. 
- Run ``` RfTrain.py``` to train the random forest regressor. 
- Run ```rfval_par.py``` to evalute the machine learning algorithm on the validation data.
- If connected with an intel realsense camera and a UR10 robot, run ```MainCode.py``` to test grasping.

### Abstract of the project:

*This report explores the initial problem formulation: "How can a robot be used for assisting activities of daily living for tetraplegic patients?". This question is then answered through a problem analysis where, the problems and needs of tetraplegic patients is researched, after which some current existing solutions for assisting them is analysed. Using the problem analysis, the fnal problem formulation: "How can a solution for grasping objects, for eating, drinking and cooking, be made using an assistive robot with an Intel RealSense D435i camera attached?" was created. To answer this, requirements for a solution were created, focusing on the reliability, speed of grasping and controlling the robot. To fulfll the requirements, point clouds, hand-eye calibration and aﬀordance are discussed. A random forrest grasp aﬀordance regressor was made with a grasp aﬀordance location success of 70% and a Support vector regressor pose estimator was formulated but not integrated into the solution. Most of the solution was tested, with the grasp action not being fully implemented. The conclusion is that the implementation is not a complete solution for assisting ADL, however a discussion is given on how to further aﬀordance studies and amend the shortcomings of the implementation.*
 
 ### Dependencies.
 - [Scikit](https://scikit-learn.org/stable/)
 - [Open3d](http://www.open3d.org/)
 - [URX python](https://github.com/SintefManufacturing/python-urx/)
 
