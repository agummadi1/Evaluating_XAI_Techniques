<h1>Evaluating XAI Techniques</h1>

<h2>Datasets</h2>

<h3>(1)  MEMS datasets:</h3>
To build these datasets, an experiment was conducted in the motor testbed to collect machine condition data (i.e., acceleration) for different health conditions. During the experiment, the acceleration signals were collected from both piezoelectric and MEMS sensors at the same time with the sampling rate of 3.2 kHz and 10 Hz, respectively, for X, Y, and Z axes. Different levels of machine health condition can be induced by mounting a mass on the balancing disk, thus different levels of mechanical imbalance are used to trigger failures. Failure condition can be classified as one of three possible states - normal, near-failure, and failure.
Multiple levels of the mechanical imbalance can be generated in the motor testbed (i.e., more masses indicate worse health condition). In this experiment, three levels of mechanical imbalance (i.e., normal, near-failure, failure) were considered
Acceleration data were collected at the ten rotational speeds (100, 200, 300, 320, 340, 360, 380, 400, 500, and 600 RPM) for each condition, while the motor is running, 50 samples were collected at 10 s interval, for each of the ten rotational speeds. We use this same data for defect-type classification and learning transfer tasks.

<h3>(2) N-BaIoT dataset:</h3>
It was created to detect IoT botnet attacks and is a useful resource for researching cybersecurity issues in the context of the Internet of Things (IoT).
This data was gathered from nine commercial IoT devices that were actually infected by two well-known botnets, Mirai and Gafgyt.

Every data instance in the dataset has access to a variety of features. These attributes are divided into multiple groups:

A. Stream Aggregation: These functions offer data that summarizes the traffic of the past few days. This group's categories comprise:
H: Statistics providing an overview of the packet's host's (IP) recent traffic.
HH: Statistics providing a summary of recent traffic from the host (IP) of the packet to the host of the packet's destination.
HpHp: Statistics providing a summary of recent IP traffic from the packet's source host and port to its destination host and port.
HH-jit: Statistics that summarize the jitter of the traffic traveling from the IP host of the packet to the host of its destination.

B. Time-frame (Lambda): This characteristic indicates how much of the stream's recent history is represented in the statistics. They bear the designations L1, L3, L5, and so forth.

C. Data Taken Out of the Packet Stream Statistics: Among these characteristics are:

Weight: The total number of objects noticed in recent history, or the weight of the stream.

Mean: The statistical mean is called the mean.

Std: The standard deviation in statistics.

Radius: The square root of the variations of the two streams.

Magnitude: The square root of the means of the two streams.

Cov: A covariance between two streams that is roughly estimated.

Pcc: A covariance between two streams that is approximated.

The dataset consists of the following 11 classes: benign traffic is defined as network activity that is benign and does not have malicious intent, and 10 of these classes represent different attack tactics employed by the Gafgyt and Mirai botnets to infect IoT devices. 

1. benign: There are no indications of botnet activity in this class, which reflects typical, benign network traffic. It acts as the starting point for safe network operations.

2. gafgyt.combo: This class is equivalent to the "combo" assault of the Gafgyt botnet, which combines different attack techniques, like brute-force login attempts and vulnerability-exploiting, to compromise IoT devices.

3. gafgyt.junk: The "junk" attack from Gafgyt entails flooding a target device or network with too many garbage data packets, which can impair operations and even result in a denial of service.

4. gafgyt.scan: Gafgyt uses the "scan" attack to search for IoT devices that are susceptible to penetration. The botnet then enumerates and probes these devices in an effort to locate and compromise them.

5. gafgyt.tcp: This class embodies the TCP-based attack of the Gafgyt botnet, which targets devices using TCP-based exploits and attacks.

6. gafgyt.udp: The User Datagram Protocol (UDP) is used in Gafgyt's "udp" assault to initiate attacks, such as bombarding targets with UDP packets to stop them from operating.

7. mirai.ack: To take advantage of holes in Internet of Things devices and enlist them in the Mirai botnet, Mirai's "ack" attack uses the Acknowledgment (ACK) packet.

8. mirai.scan: By methodically scanning IP addresses and looking for vulnerabilities, Mirai's "scan" assault seeks to identify susceptible Internet of Things (IoT) devices.

9. mirai.syn: The Mirai "syn" attack leverages vulnerabilities in Internet of Things devices to add them to the Mirai botnet by using the SYN packet, which is a component of the TCP handshake procedure.

10. mirai.udp: Based on the UDP protocol, Mirai's "udp" attack includes bombarding targeted devices with UDP packets in an attempt to interfere with their ability to function.

11. mirai.udpplain: This class represents plain UDP assaults that aim to overload IoT devices with UDP traffic, causing service disruption. It is similar to the prior "udp" attack by Mirai.

The dataset consists of data collected from 9 IoT devices, however, for this paper, we have chosen to specially work on the dataset of DEVICE 7 - Samsung SNH 1011 N Webcam which has only classes 1 -6  


<h2>How to run the program</h2>

Inside the MEMS_4_Metrics or d7_4_Metrics folder, you will find programs for each of the 5 models used in this paper. Each one of these programs outputs: 

1. The accuracy for the AI model.
2. The values for y_axis for the sparsity (that will need to be copied and pasted into the sparsity_gen_shap/lime.py).
3. Top features in importance order (that will be needed to rerun these same programs to obtain new Accuracy values for Descriptive Accuracy. Take note of the values as you use less features and input these values in desc_acc_gen_shap/lime.py ).
4. The number of samples being used.
   
Descriptive Accuracy:
To generate Descriptive Accuracy Graphs, see the code desc_acc_gen_shap/lime.py in the respective folders.

Sparsity:
To generate Sparsity Graphs, see the code sparsity_gen_shap/lime.py in the respective folders.

Stability:
For Stability metrics, run the programs 3x or more and note the obtained top k features in each run and compare.

Efficiency:
From the output file, take note of the time spent.


Robustness:
Inside the Robustness folder, firts run the code: threshold_mems.py or threshold_d7.py to generate a csv file. Then run analyze_threshold_mems_LIME/SHAP.py or analyze_threshold_d7_LIME/SHAP.py to generate the Robustness Sensitivity graph.

Completeness:
Inside the Completeness folder, run the final_mems_d7_completeness_lime/shap.py for each label, to generate the completeness graph. 
Note: According to the dataset being used, you will need to comment the code of the other dataset. They have been clearly marked by #mems or #d7


<h2>Sample Evaluation Results</h2>

<h4>Sample 1</h4>

Sample explanation




