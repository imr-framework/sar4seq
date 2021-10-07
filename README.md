---
layout: page
title: sar4seq
---

### Prerequisite:

Installed Matlab R2015a or later

Let us know if you face any problems in running the code or validating calculated values by posting in Issues.

This code implements the methods and corresponding results shown in Geethanath, S., Kabil, J. and Vaughan, J.T., 2020. Radio Frequency Safety Assessment for Open Source Pulse Sequence Programming. Safety and Biological Effects in MRI, p.207.

### Demos

Run the SAR4seq.m to:

     1. Compute RF safety metrics for Pulseq sequences 
      a. For Pulseq sequences for deployment on Siemens scanners - 
     computes time averaged RF power for the sequence
      b. For Pulseq sequences for deployment on GE scanners (via TOPPE) -
     computes the whole body SAR in W/kg
     
 
    Parameters
    ----------
       seq_path : Path to Pulseq sequence file - string
       seq : Pulseq sequence object determining system parameters - seq
       object
       Sample_weight : weight of the sample being imaged - double
 
     Returns
     -------
       Time averaged RF power : double
       Whole body SAR : double
            
  2.  Providing no inputs computes SAR for a Turbo Spin Echo sequence with default system hardware parameters and sample weight

