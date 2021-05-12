# DeepBeam data and repo

The code in this repository has been used to generate the results for the paper 

> M. Polese, F. Restuccia, and T. Melodia, "DeepBeam: Deep Waveform Learning for Coordination-Free Beam Management in mmWave Networks", Proc. of ACM Intl. Symp. on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (ACM MobiHoc), October 2021.

The associated dataset can be found [at this link](TODO).

Please reference the paper if you use the code or data from the dataset: [bibtex entry](https://ece.northeastern.edu/wineslab/wines_bibtex/polese2021mobihoc.txt)

# Dataset structure

The DeepBeam dataset can be found [at this link](TODO).

It contains 19 HDF5 files that represent a data collection campaign run on the NI mmWave Transceiver System with four SiBeam 60 GHz radio heads and on two Pi-Radio digital 60 GHz radios. The data collection campaign is described in [Section 4 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf).

The data is organized as follows.

## NI-based dataset for TXB classification

Each HDF5 file contains I/Q samples corresponding to 3 (parameter `num_gains` in the scripts) receiver gain values (40 dB, 50 dB, 60 dB) to represent three different received SNR values (i.e., in a range between -15 dB and 20 dB) and 24 TX beams or 12 TX beams. 

The files are organized using HDF5 datasets. Each file contains four datasets
- "iq" contains the I/Q samples (one column for the I samples, the other for the Q samples)
- "tx_beam" contains a label with the transmit beam used for the corresponding I/Q sample (i.e., entry N in the tx_beam dataset corresponds to entry N in the iq dataset)
- "rx_beam" contains a label with the receive beam used for the corresponding I/Q sample (i.e., entry N in the rx_beam dataset corresponds to entry N in the iq dataset)
- "gain" contains a label with the receiver gain value for the corresponding I/Q sample (i.e., entry N in the gain dataset corresponds to entry N in the iq dataset)

The total number of entries in each dataset depends on whether 24 or 12 TX beams are used (parameter `num_beams` in the scripts).  For each `(gain, tx_beam)` pair, we collected 10000 frames (parameter `num_frames_for_gain_tx_beam_pair` in the scripts). Each frame contains 15 blocks (parameter `num_blocks_per_frame` in the scripts). Each frame contains 2048 I/Q samples (parameter `num_samples_per_block` in the scripts). Therefore, the total number of entries is `num_gains * num_beams * num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block`.

The I/Q samples are arranged sequentially, according to the following logic:

```
For gain in [40, 50, 60]:
    For tx_beam in 0:num_beams:
        Store num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block of the (gain, tx_beam) pair
```

The receive beam is fixed (boresight of the antenna array).

#### Basic configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the basic configuration, we provide the 24 and 12 TX beams data sets with 4 different configurations of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-basic-config-24-beams-tx-ant-0-rx-ant-1.h5`
  - TX antenna 1, RX antenna 0 `srf-basic-config-24-beams-tx-ant-1-rx-ant-0.h5`
  - TX antenna 2, RX antenna 1 `srf-basic-config-24-beams-tx-ant-2-rx-ant-1.h5`
  - TX antenna 3, RX antenna 1 `srf-basic-config-24-beams-tx-ant-3-rx-ant-1.h5`
 
- 12 TX beams
  - TX antenna 0, RX antenna 1 `srf-basic-config-12-beams-tx-ant-0-rx-ant-1.h5`
  - TX antenna 1, RX antenna 0 `srf-basic-config-12-beams-tx-ant-1-rx-ant-0.h5`
  - TX antenna 2, RX antenna 1 `srf-basic-config-12-beams-tx-ant-2-rx-ant-1.h5`
  - TX antenna 3, RX antenna 1 `srf-basic-config-12-beams-tx-ant-3-rx-ant-1.h5`

#### Diagonal configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the diagonal configuration, we provide the 24 and 12 TX beams data sets with one configuration of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-diagonal-config-24-beams-tx-ant-0-rx-ant-1.h5`
 
- 12 TX beams
  - TX antenna 0, RX antenna 1 `srf-diagonal-config-12-beams-tx-ant-0-rx-ant-1.h5`
  
#### Obstacle configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the obstacle configuration, we provide the 24 and 12 TX beams data sets with one configuration of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-obstacle-config-24-beams-tx-ant-0-rx-ant-1.h5`
 
- 12 TX beams
  - TX antenna 0, RX antenna 1 `srf-obstacle-config-12-beams-tx-ant-0-rx-ant-1.h5`  
  

## Pi-Radio-based dataset for TXB classification

A single HDF5 file is available for the Pi-Radio TXB data set. It contains I/Q samples corresponding to31 (parameter `num_gains` in the scripts) transmitter gain values and 5 TX beams (parameter `num_beams` in the scripts). 

The files are organized using HDF5 datasets. Each file contains four datasets
- "iq" contains the I/Q samples (one column for the I samples, the other for the Q samples)
- "tx_beam" contains a label with the transmit beam used for the corresponding I/Q sample (i.e., entry N in the tx_beam dataset corresponds to entry N in the iq dataset)
- "rx_beam" contains a label with the receive beam used for the corresponding I/Q sample (i.e., entry N in the rx_beam dataset corresponds to entry N in the iq dataset)
- "gain" contains a label with the receiver gain value for the corresponding I/Q sample (i.e., entry N in the gain dataset corresponds to entry N in the iq dataset)

For each `(gain, tx_beam)` pair, we collected 10000 frames (parameter `num_frames_for_gain_tx_beam_pair` in the scripts). Each frame contains 5 blocks (parameter `num_blocks_per_frame` in the scripts). Each frame contains 2048 I/Q samples (parameter `num_samples_per_block` in the scripts). Therefore, the total number of entries is `num_gains * num_beams * num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block`.

The I/Q samples are arranged sequentially, according to the following logic:

```
For gain in ['att-tx-0-0-', 'att-tx-5-0-', 'att-tx-5-4-']: # these values correspond to increasing attenuation values
    For tx_beam in 0:num_beams:
        Store num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block of the (gain, tx_beam) pair
```

The receive beam is fixed (boresight of the antenna array).

The Pi-Radio-based dataset is in the file `mrf-basic-config-5-beams.h5` for the configuration shown in [Figure 8 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf).

## NI-based dataset for AoA classification

Each HDF5 file contains I/Q samples corresponding to 3 (parameter `num_gains` in the scripts) receiver gain values (40 dB, 50 dB, 60 dB) to represent three different received SNR values (i.e., in a range between -15 dB and 20 dB), 3 TX beams for the 24 TX beams codebook, and 3 AoA values (-45, 0, 45) (parameter `num_angles` in the scripts). 

The files are organized using HDF5 datasets. Each file contains four datasets
- "iq" contains the I/Q samples (one column for the I samples, the other for the Q samples)
- "tx_beam" contains a label with the transmit beam used for the corresponding I/Q sample (i.e., entry N in the tx_beam dataset corresponds to entry N in the iq dataset)
- "rx_beam" contains a label with the receive beam used for the corresponding I/Q sample (i.e., entry N in the rx_beam dataset corresponds to entry N in the iq dataset)
- "gain" contains a label with the receiver gain value for the corresponding I/Q sample (i.e., entry N in the gain dataset corresponds to entry N in the iq dataset)
- "angle" contains a label with the receiver angle value for the corresponding I/Q sample (i.e., entry N in the angle dataset corresponds to entry N in the iq dataset)

In this case, `num_beams` is 3. For each `(gain, tx_beam, angle)` tuple, we collected 10000 frames (parameter `num_frames_for_gain_tx_beam_pair` in the scripts). Each frame contains 15 blocks (parameter `num_blocks_per_frame` in the scripts). Each frame contains 2048 I/Q samples (parameter `num_samples_per_block` in the scripts). Therefore, the total number of entries is `num_gains * num_beams * num_angles * num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block`.

The I/Q samples are arranged sequentially, according to the following logic:

```
For gain in [40, 50, 60]:
    For tx_beam in [4, 12, 20]:
        For angle in [-45, 0, 45]:
            Store num_frames_for_gain_tx_beam_pair * num_blocks_per_frame * num_samples_per_block of the (gain, tx_beam, angle) tuple
```

The receive beam is fixed (boresight of the antenna array).

#### Basic configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the basic configuration, we provide the AoA data sets with 4 different configurations of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-basic-config-24-beams-aoa-tx-ant-0-rx-ant-1.h5`
  - TX antenna 1, RX antenna 0 `srf-basic-config-24-beams-aoa-tx-ant-1-rx-ant-0.h5`
  - TX antenna 2, RX antenna 1 `srf-basic-config-24-beams-aoa-tx-ant-2-rx-ant-1.h5`
  - TX antenna 3, RX antenna 1 `srf-basic-config-24-beams-aoa-tx-ant-3-rx-ant-1.h5`

#### Diagonal configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the diagonal configuration, we provide the AoA data sets with one configuration of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-diagonal-config-24-beams-aoa-tx-ant-0-rx-ant-1.h5`

#### Obstacle configuration (see [Figure 7 of the DeepBeam paper](https://arxiv.org/pdf/2012.14350.pdf))

For the obstacle configuration, we provide the AoA data sets with one configuration of the SiBeam 60 GHz heads:

- 24 TX beams
  - TX antenna 0, RX antenna 1 `srf-obstacle-config-24-beams-aoa-tx-ant-0-rx-ant-1.h5`
 


# Requirements

TODO add requirements.txt

# Source code structure

The 

