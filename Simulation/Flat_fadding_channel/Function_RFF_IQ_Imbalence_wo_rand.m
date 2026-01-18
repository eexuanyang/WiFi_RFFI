function [Output_Waveform, I_Offset_Value, Q_Offset_Value, IQ_Gain_Imbalance_Value, IQ_Phase_Shift_Imbalance_Value] = Function_RFF_IQ_Imbalence_wo_rand(Input_Waveform, IQ_Offset_Max, IQ_Gain_Imbalance_Max, IQ_Phase_Shift_Imbalance_Max, Simulation_Seed)

    if IQ_Offset_Max ~= 0
        I_Offset_Value = mean(abs(real(Input_Waveform(1:floor(size(Input_Waveform,1)/10)))))*(10^(IQ_Offset_Max/20));
    else
        I_Offset_Value = 0;
    end
    Q_Offset_Value = I_Offset_Value;
    Output_Waveform = iqimbal(Input_Waveform, IQ_Gain_Imbalance_Max, IQ_Phase_Shift_Imbalance_Max)...
                        + complex(I_Offset_Value, Q_Offset_Value);
    IQ_Gain_Imbalance_Value = IQ_Gain_Imbalance_Max;
    IQ_Phase_Shift_Imbalance_Value = IQ_Phase_Shift_Imbalance_Max;