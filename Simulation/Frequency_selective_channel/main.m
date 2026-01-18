close all; 
clear all; 
clc

long_subca = [7:32 34:59];

legd_size = 18;
gca_size = 23;
legd_size_large = 25;
gca_size_large = 27;

%% Channel
load("lltf_fre_tx1_rx1_channel.mat")
tx1_rx1_channel = abs(lltf_fre);

load("lltf_fre_tx2_rx1_channel.mat")
tx2_rx1_channel = abs(lltf_fre);

load("lltf_fre_tx1_rx2_channel.mat")
tx1_rx2_channel = abs(lltf_fre);

load("lltf_fre_tx2_rx2_channel.mat")
tx2_rx2_channel = abs(lltf_fre);

tx1_rx1_channel = tx1_rx1_channel(long_subca);
tx2_rx1_channel = tx2_rx1_channel(long_subca);
tx1_rx2_channel = tx1_rx2_channel(long_subca);
tx2_rx2_channel = tx2_rx2_channel(long_subca);

data = [tx1_rx1_channel; tx2_rx1_channel; tx1_rx2_channel; tx2_rx2_channel];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

ax = gca;
ax.Units = 'normalized';  
ax.Position = [0.15, 0.3, 0.5, 0.4]; 

ax.DataAspectRatio = [1, 1, 1];  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(tx1_rx1_channel,'-','color',[0.4660 0.6740 0.1880],LineWidth=3)
hold on
plot(tx2_rx1_channel,'r-.',LineWidth=3)
hold on
plot(tx1_rx2_channel,'b--',LineWidth=3)
hold on
plot(tx2_rx2_channel,'k:',LineWidth=3)
% legend("T1-R1","T1-R2","T2-R1","T2-R2", 'Fontsize', legd_size_large)

lgd = legend("T1-R1","T1-R2","T2-R1","T2-R2", ...
    'FontSize', legd_size_large, ...
    'NumColumns', 2);

% lgd.Location = 'northoutside'; 

xlabel({'Subcarrier', '(a)'})
ylabel("Amplitude")
xlim([0 53])
ylim([min(data) max(data)])
xticks([0 10 20 30 40 50])
set(gca,'FontSize',gca_size_large,'FontName', 'Times New Roman');

%% RFF feature

load("lltf_fre_tx1_rx1_filter.mat")
tx1_rx1_data = abs(lltf_fre);
load("ht_ltf_fre_tx1_rx1_filter.mat")
tx1_rx1_data_ht = abs(ht_ltf_fre);

load("lltf_fre_tx2_rx1_filter.mat")
tx2_rx1_data = abs(lltf_fre);
load("ht_ltf_fre_tx2_rx1_filter.mat")
tx2_rx1_data_ht = abs(ht_ltf_fre);

load("lltf_fre_tx1_rx2_filter.mat")
tx1_rx2_data = abs(lltf_fre);
load("ht_ltf_fre_tx1_rx2_filter.mat")
tx1_rx2_data_ht = abs(ht_ltf_fre);

load("lltf_fre_tx2_rx2_filter.mat")
tx2_rx2_data = abs(lltf_fre);
load("ht_ltf_fre_tx2_rx2_filter.mat")
tx2_rx2_data_ht = abs(ht_ltf_fre);

t1_feature1 = tx1_rx1_data_ht(long_subca)./tx1_rx1_data(long_subca);
t1_feature2 = tx1_rx2_data_ht(long_subca)./tx1_rx2_data(long_subca);

t2_feature1 = tx2_rx1_data_ht(long_subca)./tx2_rx1_data(long_subca);
t2_feature2 = tx2_rx2_data_ht(long_subca)./tx2_rx2_data(long_subca);

t1_feature1 = t1_feature1./sqrt(sum(t1_feature1.^2));
t1_feature2 = t1_feature2./sqrt(sum(t1_feature2.^2));
t2_feature1 = t2_feature1./sqrt(sum(t2_feature1.^2));
t2_feature2 = t2_feature2./sqrt(sum(t2_feature2.^2));

data = [t1_feature1; t1_feature2; t2_feature1; t2_feature2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]);

ax = gca;
ax.Units = 'normalized';  
ax.Position = [0.15, 0.3, 0.5, 0.4];  

ax.DataAspectRatio = [1, 1, 1]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(t1_feature1,'-','color',[0.4660 0.6740 0.1880],LineWidth=3)
hold on
plot(t1_feature2,'r-.',LineWidth=3)
hold on
plot(t2_feature1,'b--',LineWidth=3)
hold on
plot(t2_feature2,'k:',LineWidth=3)
% legend("T1-R1","T1-R2","T2-R1","T2-R2", 'Fontsize', legd_size_large)
lgd = legend("T1-R1","T1-R2","T2-R1","T2-R2", ...
    'FontSize', legd_size_large, ...
    'NumColumns', 2);

% lgd.Location = 'northoutside';  

xlabel({'Subcarrier', '(b)'})
ylabel("Amplitude")
yticks([0.1 0.2])
xlim([0 53])
ylim([min(data) max(data)])
xticks([0 10 20 30 40 50])
set(gca,'FontSize',gca_size_large,'FontName', 'Times New Roman');