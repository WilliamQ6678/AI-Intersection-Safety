Using AI Computer Vision for Real-Time Detection of Safe Left-Turning and Crossing Conditions at Dangerous Low-Visibility Intersections 

Abstract:

Intersections with limited visibility can create dangerous situations for drivers and pedestrians, particularly when parked vehicles, vegetation, fences, or other obstructions to prevent drivers from clearly seeing approaching traffic. This project proposes an AI-powered intersection safety system designed to provide real-time visual warnings based on detected traffic conditions. A camera-based computer vision system will monitor the intersection and identify vehicles, pedestrians, and their movements. The system will analyze the positions and trajectories of detected objects to determine whether a straight or left-turn movement can be made safely. Based on this assessment, an external signal will display a yellow flashing light to indicate caution or a red light when the system determines that proceeding is unsafe. The system will be evaluated using recorded and controlled intersection scenarios to measure detection of accuracy, response time, and the reliability of its safety classifications. By combining real-time object detection, tracking, and decision-making with a physical warning signal, this project aims to investigate whether computer vision can provide an additional layer of safety at intersections where conventional traffic infrastructure or visibility is limited. 

Methods:

The system will use computer vision to detect and track vehicles approaching the intersection. For each detected vehicle, its position will be estimated using the camera's calibrated field of view, and its speed will be calculated from changes in position across consecutive video frames. Perspective calibration will be used to convert pixel-based movement into approximate real-world distance and velocity measurements.

Using the estimated position and velocity of each vehicle, the system will predict the time at which each vehicle will reach the relevant conflict region of the intersection. The system will calculate Post-Encroachment Time (PET) for pairs of vehicles whose predicted paths intersect. PET represents the time difference between one vehicle leaving a conflict region and another vehicle entering it. Lower PET values indicate that vehicles are passing through the conflict region closer together in time and therefore represent greater potential collision risk.

For each intended left-turn or straight-through movement, the system will evaluate potential conflicts from both relevant directions of approaching traffic. PET will be calculated for each potential conflicting vehicle, and the minimum PET will be selected as the critical value. This ensures that the safety assessment is determined by the vehicle predicted to reach the conflict region first, rather than by an average or less immediate conflict.

The minimum PET will then be compared with the estimated time required for the subject vehicle to complete its intended maneuver plus a predetermined safety buffer:

(minimum) PET > (manuever) T + (buffer) T

If this condition is satisfied, the movement will be classified as safe. If the minimum PET is less than or equal to the required maneuver time plus the safety buffer, the movement will be classified as unsafe, triggering the appropriate warning signal.

The system will be evaluated using recorded and controlled intersection scenarios with varying vehicle speeds, traffic volumes, and visibility conditions. Performance will be measured using detection accuracy, tracking accuracy, PET estimation accuracy, response time, and the accuracy of the resulting safe/unsafe classifications.

