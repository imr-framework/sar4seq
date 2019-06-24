---
layout: page
title: sar4seq
---

### Specific Absorption Rate (SAR)

Check out a video on [YouTube](https://www.youtube.com/watch?v=2XroYwUxzD4) to learn more about SAR. 
To learn more about SAR and MR safety, please visit the [MR safety page](https://www.youtube.com/watch?v=s1x37l8xWzk)


### Why sar4seq?
The advent of open source pulse sequence programming has enabled a multitude of opportunities for rapid prototyping of MR acquisition methods. For more, read [here](https://wiki.opensourceimaging.org/Methods#Pulse_Sequence_Programming)

It is critical to evaluate sequences designed using open source tools such as [pypulseq](https://imr-framework.github.io/pages/pypulseq.html) for [MR safety](https://imr-framework.github.io/pages/mrsafety.html). This involves the computation of [local and global SAR](https://en.wikipedia.org/wiki/Specific_absorption_rate) that is MRI pulse sequence and model dependent among other factors. sar4seq provides local and global SAR values for a pulse sequence that you have developed. This will provide an important indication of the RF safety of your sequence. sar4seq can be easily built, rebuilt and deployed as it leverages the Pulseq file standard [1, 2] and Q-matrix computation [3]

In no event shall the authors be liable for any claim, damages or
other liability, whether in an action of contract, tort or otherwise, arising from, out of or
in connection with the Software or the use or other dealings in the Software.

Coming SOON! You can request SAR values for your MR pulse sequences designed with pypulseq [https://imr-framework.github.io/pages/pypulseq.html] or pulseq [http://pulseq.github.io/] to be used as online service, fill out this [form](). Read more [here](#demos).



### Demos
sar4seq demos can be found [here](). In particular, these contain example evaluations for the [Gradient Recalled Echo](http://www.mriquestions.com/gradient-echo.html) and the [Spin Echo](http://mriquestions.com/spin-echo1.html) sequences.

#### sar4seq - as an Online Service - COMING SOON! 
If you choose not to download the entire repo and run the source code, you can upload your **.seq** file to obtain the SAR values by filling out this [form](https://goo.gl/forms/1FpGeH7S9SJbaBP53).

### References
[1] [Layton, Kelvin J., et al. "Pulseq: A rapid and hardware‐independent pulse sequence prototyping framework." Magnetic resonance in medicine 77.4 (2017): 1544-1552.](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26235)

[2] [Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018): 9-15.](https://www.sciencedirect.com/science/article/pii/S0730725X1830033X)

[3] [Graesslin, Ingmar, Hanno Homann, Sven Biederer, Peter Börnert, Kay Nehrke, Peter Vernickel, Giel Mens, Paul Harvey, and Ulrich Katscher. "A specific absorption rate prediction concept for parallel transmission MR." Magnetic resonance in medicine 68, no. 5 (2012): 1664-1674.](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24138)


