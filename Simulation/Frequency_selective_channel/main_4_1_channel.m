% fitCoefMatMem_rx4: Polynomial coefficients generated using the example at
% https://ww2.mathworks.cn/help/comm/ref/comm.dpdcoefficientestimator-system-object.html
% via openExample('comm/PreDistortPowerAmplifierInputSignalExample')

close all; 
clear all; 
clc

tx_iq = 0;
tx_pa = 0;
cn_flag = 1;
rx_pa = 0;
rx_iq = 0;
noise_flag = 1;
packetSNR = 40;

tx1_filter = [1     0       0.8     0       0.5     0       0       0.2];
tx2_filter = [1     0.6     0       0.3     0       0.2     0       0.1];
rx1_filter = [1     0.8     0.3     0       0.1     0       0       0];

tx_attenu_para = 10;
rx_attenu_para = 10;

LSTF = [1 160];
LLTF = [161 320];
LSIG = [321 400];

cp_par = 0.75;

HT_SIG1 = [401 480];
HT_SIG2 = [481 560];
HT_STF = [561 640];
HT_LTF = [641 720];
HT_LTF2= [721 800];
HT_LTF3= [801 880];

%% 802.11n Packet Error Rate Simulation for 2x2 TGn Channel
%% Introduction
%% Waveform Configuration
cfgHT = wlanHTConfig;
cfgHT.ChannelBandwidth = 'CBW20'; % 20 MHz channel bandwidth
cfgHT.NumTransmitAntennas = 1;    % 2 transmit antennas
cfgHT.NumSpaceTimeStreams = 1;    % 2 space-time streams
cfgHT.PSDULength = 1000;          % PSDU length in bytes
cfgHT.MCS = 7;                   % 2 spatial streams, 64-QAM rate-5/6
cfgHT.ChannelCoding = 'BCC';      % BCC channel coding

%% Channel Configuration
tgnChannel = wlanTGnChannel;
tgnChannel.DelayProfile = 'Model-E';
tgnChannel.NumTransmitAntennas = cfgHT.NumTransmitAntennas;
tgnChannel.NumReceiveAntennas = 1;
tgnChannel.TransmitReceiveDistance = 10; % Distance in meters for NLOS
tgnChannel.LargeScaleFadingEffect = 'None';
tgnChannel.NormalizeChannelOutputs = false;

%% Simulation Parameters
snr = 45;

maxNumPEs = 1; % The maximum number of packet errors at an SNR point
maxNumPackets = 1; % The maximum number of packets at an SNR point

% Set the remaining variables for the simulation.

% Get the baseband sampling rate
fs = wlanSampleRate(cfgHT);

% Get the OFDM info
ofdmInfo = wlanHTOFDMInfo('HT-Data',cfgHT);

% Set the sampling rate of the channel
tgnChannel.SampleRate = fs;

% Indices for accessing each field within the time-domain packet
ind = wlanFieldIndices(cfgHT);

%% Processing SNR Points
S = numel(snr);
packetErrorRate = zeros(S,1);
Simulation_Seed = 1;

for i = 1:S % Use 'for' to debug the simulation
    numPacketErrors = 0;
    n = 1; % Index of packet transmitted
    while numPacketErrors<=maxNumPEs && n<=maxNumPackets
        % Generate a packet waveform
        txPSDU = randi([0 1],cfgHT.PSDULength*8,1); % PSDULength in bytes
        tx = wlanWaveformGenerator(txPSDU,cfgHT);
        
        % Add trailing zeros to allow for channel filter delay
        tx = [zeros(200,1); tx; zeros(15,cfgHT.NumTransmitAntennas)]; %#ok<AGROW>

        %% Tx IQ DC offset and IQ imbalance
        if tx_iq == 1
            IQ_Offset_Max = -34;                           
            IQ_Gain_Imbalance_Max = 0.3;                   
            IQ_Phase_Shift_Imbalance_Max = 2;            
        else 
            IQ_Offset_Max = 0.00;                           
            IQ_Gain_Imbalance_Max = 0.;                 
            IQ_Phase_Shift_Imbalance_Max = 0.0;            
        end

        Input_Waveform = tx;
        [Output_Waveform, ~, ~, ~, ~] = Function_RFF_IQ_Imbalence_wo_rand(Input_Waveform, IQ_Offset_Max, IQ_Gain_Imbalance_Max, IQ_Phase_Shift_Imbalance_Max, Simulation_Seed);

        %% Tx PA nonlinearity
        Input_Waveform = Output_Waveform;
        
        modType = 'memPoly';
        load("fitCoefMatMem_rx4.mat");

        if tx_pa == 1
            Input_Waveform = Input_Waveform/tx_attenu_para;
            Output_Waveform = conv(tx2_filter, Input_Waveform); 
%             Output_Waveform = helperPACharMemPolyModel('signalGenerator', ...
%               Input_Waveform.', fitCoefMatMem_rx4, modType);
        else
            Output_Waveform = Input_Waveform;
        end

        %% Channel
        Input_Waveform = Output_Waveform;
        if cn_flag == 1
%             reset(tgnChannel); % Reset channel for different realization
            rng(4)
            rx = tgnChannel(Input_Waveform);
        else
            rx = Input_Waveform;
        end

        %% Rx PA nonlinearity
        Input_Waveform = rx;
        
        modType = 'memPoly';
        load("fitCoefMatMem_rx4.mat");
 
        if rx_pa == 1
            Input_Waveform = Input_Waveform/rx_attenu_para;
            Output_Waveform = helperPACharMemPolyModel('signalGenerator', ...
              Input_Waveform.', fitCoefMatMem_rx4, modType);
        else
            Output_Waveform = Input_Waveform;
        end

        %% Rx IQ DC offset and IQ imbalance
        Input_Waveform = Output_Waveform;
        if rx_iq == 1
            IQ_Offset_Max = -30;                          
            IQ_Gain_Imbalance_Max = 0.5;                 
            IQ_Phase_Shift_Imbalance_Max = 1.5;           
        else 
            IQ_Offset_Max = 0.00;                                      
            IQ_Gain_Imbalance_Max = 0.;                 
            IQ_Phase_Shift_Imbalance_Max = 0.0;         
        end

        [Output_Waveform, ~, ~, ~, ~] = Function_RFF_IQ_Imbalence_wo_rand(Input_Waveform, IQ_Offset_Max, IQ_Gain_Imbalance_Max, IQ_Phase_Shift_Imbalance_Max, Simulation_Seed);

        %% Adding noise
        Input_Waveform = Output_Waveform;
        if noise_flag == 1
            sig_power = sum((abs(Input_Waveform(201:1200))).^2) / 1000;
            noise_power = sig_power / (10^(packetSNR/10));
            rng(7)
            Output_Waveform = Input_Waveform + normrnd(0,sqrt(noise_power/2), length(Input_Waveform), 1) + 1i * normrnd(0,sqrt(noise_power/2), length(Input_Waveform), 1);
        else
            Output_Waveform = Input_Waveform;
        end

        %% Preprocessing
        rx = Output_Waveform;

        % Packet detect and determine coarse packet offset
%         coarsePktOffset = wlanPacketDetect(rx,cfgHT.ChannelBandwidth);
        coarsePktOffset = 200;

        if isempty(coarsePktOffset) % If empty no L-STF detected; packet error
            numPacketErrors = numPacketErrors+1;
            n = n+1;
            continue; % Go to next loop iteration
        end
        
        % Extract L-STF and perform coarse frequency offset correction
        lstf = rx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:); 
        coarseFreqOff = wlanCoarseCFOEstimate(lstf,cfgHT.ChannelBandwidth);
        rx = frequencyOffset(rx,fs,-coarseFreqOff);
        
        % Extract the non-HT fields and determine fine packet offset
        nonhtfields = rx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:); 
        finePktOffset = wlanSymbolTimingEstimate(nonhtfields,...
            cfgHT.ChannelBandwidth);
        
        % Determine final packet offset
        pktOffset = coarsePktOffset+finePktOffset;

        % If packet detected outwith the range of expected delays from the

        % Extract L-LTF and perform fine frequency offset correction
        lltf = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:); 
        fineFreqOff = wlanFineCFOEstimate(lltf,cfgHT.ChannelBandwidth);
        rx = frequencyOffset(rx,fs,-fineFreqOff);

        lltf = rx(pktOffset+(LLTF(1):LLTF(2)),:);
        lltf_fre = fftshift(fft(lltf(round(32*cp_par):round(32*cp_par)+64-1)));

        interval_para = 1;
        lltf1_fre_a = fftshift(fft(lltf(round(32*cp_par)+interval_para:round(32*cp_par)+64-1+interval_para)));

        lltf1_fre_a = lltf1_fre_a./sqrt(sum(lltf1_fre_a.^2));

        ht_ltf = rx(pktOffset+(HT_LTF(1):HT_LTF(2)),:);
        ht_ltf_fre = fftshift(fft(lltf(round(16*cp_par):round(16*cp_par)+64-1)));

        figure
        subplot(211)
        plot(abs(lltf_fre(7:59)),LineWidth=1)
        subplot(212)
        plot(abs(lltf1_fre_a(7:59)-lltf_fre(7:59)),LineWidth=1)
        n = n+1;

        save("lltf_fre_tx2_rx2_channel.mat","lltf_fre");
    end
end
