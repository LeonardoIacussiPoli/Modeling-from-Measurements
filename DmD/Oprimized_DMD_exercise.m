clear all
close all
clc

%% LOAD DATA
t = 0:2:60-2;
t_real = t+1845;
SH = [20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,100,92,70,10,11,137,137,18,22,52,83,18,10,9,65];
CL = [32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,34,45,40,15,15,60,80,26,18,37,50,35,12,12,25];

X = [SH ; CL];

%%% spline interpolation before DMD
spline_flag = false;

if (spline_flag == true)
    t_old = t;
    t = 0:0.1:58;
    SH = spline(t_old,SH,t);
    CL = spline(t_old,CL,t);
    X = [SH ; CL];
end

%%
% split the initial dataset into train dataset and test dataset
train_dimension  = (length(t)*2/3); % number of samples for the training dataset

X_train = X(:,1:train_dimension);
X_test = X(:,train_dimension+1:end);
t_train = t(1:train_dimension);

% target rank
r = 2;
proxfun = @(alpha) min(real(alpha),0) + 1i*imag(alpha);

%opts = varpro_opts('maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);
opts = varpro_opts();

imode = 1;
[w,e,b] = optdmd(X_train,t_train,r,imode,opts,[],[],[],[],[]);

X_predicted = real(w*diag(b)*exp(e*t));

[sorted, idx] = sort(imag(e));
e_sorted = e(idx);
w_sorted = w(:,idx);
b_sorted = b(idx);

time_delay = 30;
t_pred = t+time_delay;
%%
figure(1)
subplot(4,1,1);
title('Predator-prey population prediction with Opt-DMD','FontSize',15)
hold on
patch([1845 t_train(end)+1845 t_train(end)+1845 1845], [500 500 -500 -500], [0.7 0.7 0.7], EdgeColor = 'none')
hold off
alpha(0.3)
hold all
plot(t+1845,X(1,:),'--b', LineWidth=2)
plot(t+1845,X_predicted(1,:),'b', LineWidth=2);
plot(t+1845,X(2,:),'--r', LineWidth=2)
plot(t+1845,X_predicted(2,:),'r', LineWidth=2);
hold off
ylim([-10 180])
xlim([1845 1845+60])
legend('Training dataset','Snowshoe hare','Snowshoe hare prediction','Canadian lynx','Canadian lynx predicted', 'FontSize', 15)
xlabel('Time [years]', 'FontSize', 20)
ylabel('Number of animals', 'FontSize', 20)
ax = gca; 
ax.FontSize = 13; 
grid on


%% BAGGING
n = length(X_train);
k=n-floor(n*0.70);

lambda0 = 1.0e10;
maxlam = 100e4;
lamup = 1.0e4;
lamdown = 1.0e1;
maxiter = 100;
tol = 1.0e-10;
oeps_stall = 1.0e-12;
gamma = 1e4;

% let's set some optimization parameters
%opts = varpro_opts('lambda0',lambda0,'maxlam',maxlam,'lamup',lamup,'lamdown',lamdown,'maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);
opts = varpro_opts();

number_of_bagging =400;
e_total = zeros([r,number_of_bagging]);
b_total = zeros([r,number_of_bagging]);
w_total = zeros([2,r,number_of_bagging]);

time_series_matrix = zeros([2,length(t), number_of_bagging]);

for ii=1:number_of_bagging
    % randomly select 
    index = randperm(n,k);
    index = sort(index);
    X_red = X_train(:,index);
    t_red = t(index);
    e_init = e;
    [w,e,b] = optdmd(X_red,t_red,r,imode, opts, e_init,[],[],[],[]);
    
    % sort eigen-values by imaginary part
    [sorted_imag, idx_imag] = sort(imag(e));

    time_series_matrix(:,:,ii) = real(w*diag(b)*exp(e*t));
    
    if (any(b> 70))
        e_total(:,ii) = nan+0i;
        b_total(:,ii) = nan;
        w_total(:,:,ii) = nan+0i;
    else
        e_total(:,ii) = e(idx_imag);
        b_total(:,ii) = b(idx_imag);
        w_total(:,:,ii) = w(:,idx_imag);
    end
end

% REMOVE NAN 
e_total_no_nan = ones([2,number_of_bagging])+1i;
b_total_no_nan = ones([2,number_of_bagging]); 
w_total_no_nan = ones([2,2,number_of_bagging])+1i;
j=1;
for ii=1:number_of_bagging
    if (sum(isnan(b_total(:,ii))) == 0)
        e_total_no_nan(:,j) = e_total(:,ii);
        b_total_no_nan(:,j) = b_total(:,ii);
        w_total_no_nan(:,:,j) = w_total(:,:,ii);
        j=j+1;
    end
end

e_total = e_total_no_nan(:,1:j-1);
b_total = b_total_no_nan(:,1:j-1);
w_total = w_total_no_nan(:,:,1:j-1);

%%


% mean value calculation
% time_series_MeanValue = mean(time_series_matrix,3);
% time_series_upper_boud = prctile(time_series_matrix,95,3);
% time_series_lower_boud = prctile(time_series_matrix,5,3);

e_mean = mean(e_total, 2);
b_mean = mean(b_total, 2);
w_mean = mean(w_total, 3);

time_series_MeanValue = real(w_mean*diag(b_mean)*exp(e_mean*t));

subplot(4,1,2);
title('Predator-prey population prediction with BOP-DMD','FontSize',15)
hold on
patch([1845 t_train(end)+1845 t_train(end)+1845 1845], [500 500 -500 -500], [0.7 0.7 0.7], EdgeColor = 'none')
hold off
alpha(0.3)
hold all
plot(t+1845,X(1,:),'--b',LineWidth=2)
plot(t+1845,time_series_MeanValue(1,:),'b', LineWidth=2);
plot(t+1845,X(2,:),'--r',LineWidth=2)
plot(t+1845,time_series_MeanValue(2,:),'r', LineWidth=2);
hold off
% hold on
% 
% t_suport = [t+1845, fliplr(t+1845)];
% inBetween = [time_series_lower_boud, fliplr(time_series_upper_boud)];
% fill(t_suport, inBetween, [0.4660 0.6740 0.1880],EdgeColor = 'none');
% alpha(0.3)
% hold off

ylim([-10 180])
xlim([1845 1845+60])
legend('Training dataset','Snowshoe hare','Snowshoe hare prediction','Canadian lynx','Canadian lynx predicted','FontSize', 15)
xlabel('Time [years]', 'FontSize', 20)
ylabel('Number of animals', 'FontSize', 20)
ax = gca; 
ax.FontSize = 13; 
grid on


%% TIME DELAYED DMD 
number_of_time_delays = 3;

X_time_delays = zeros([(number_of_time_delays+1)*2,length(X_train)-number_of_time_delays]);

for ii=1:(number_of_time_delays+1)
    X_time_delays(ii,:) = SH(ii:length(X_train)-number_of_time_delays-1+ii);
    X_time_delays(ii+number_of_time_delays+1,:) = CL(ii:length(X_train)-number_of_time_delays-1+ii);
end

%opts = varpro_opts();


% 1 --- fit to unprojected data
imode = 1;
r = (number_of_time_delays+1)*2;
[w,e,b] = optdmd(X_time_delays,t(1:length(t_train)-number_of_time_delays),r,imode, opts, [],[],[],[],[]);
x_td = real(w*diag(b)*exp(e*t));
%%
% PLOT 
subplot(4,1,3);
title('Predator-prey population prediction with Time Delay Opt-DMD','FontSize',15)
hold on
patch([1845 t_train(end)+1845 t_train(end)+1845 1845], [500 500 -500 -500], [0.7 0.7 0.7], EdgeColor = 'none')
hold off
alpha(0.3)
hold all
plot(t+1845,X(1,:),'--b', LineWidth=2)
plot(t+1845,x_td(1,:),'b', LineWidth=2);
plot(t+1845,X(2,:),'--r', LineWidth=2)
plot(t+1845,x_td(number_of_time_delays*2-1,:),'r', LineWidth=2);
hold off
ylim([-10 180])
xlim([1845 1845+60])
legend('Training dataset','Snowshoe hare','Snowshoe hare prediction','Canadian lynx','Canadian lynx predicted', 'FontSize', 15)
xlabel('Time [years]', 'FontSize', 20)
ylabel('Number of animals', 'FontSize', 20)
ax = gca; 
ax.FontSize = 13; 
grid on
















%% TIME DELAY DMD WITH BAGGING

% lambda0 = 1.0e10;
% maxlam = 500;
% lamup = 1000;
% lamdown = 1000;
% maxiter = 100;
% tol = 1.0e-8;
% eps_stall = 1.0e-16;
%opts = varpro_opts('lambda0',lambda0,'maxlam',maxlam,'lamup',lamup,'lamdown',lamdown,'maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);
%opts = varpro_opts();

n = length(X_train)-number_of_time_delays;
k=n-floor(n*0.10);

number_of_bagging =100;
time_series_matrix = zeros([(number_of_time_delays+1)*2,length(t), number_of_bagging]);
r = (number_of_time_delays+1)*2;
e_total = zeros([r,number_of_bagging]);
b_total = zeros([r,number_of_bagging]);
w_total = zeros([(number_of_time_delays+1)*2,r,number_of_bagging]);


for ii=1:number_of_bagging
    index = randperm(n,k);
    index = sort(index);
    
    X_time_delays = zeros([(number_of_time_delays+1)*2,length(X_train)-number_of_time_delays]);
    
    for j=1:(number_of_time_delays+1)
        X_time_delays(j,:) = SH(j:length(X_train)-number_of_time_delays-1+j);
        X_time_delays(j+number_of_time_delays+1,:) = CL(j:length(X_train)-number_of_time_delays-1+j);
    end

    X_red = X_time_delays(:,index);
    e_init = e;
    t_red = t(index);
    [w,e,b] = optdmd(X_red,t_red,r,imode, opts, e_init,[],[],[],[]);
    
    [sorted_imag, idx] = sort(imag(e));

    e = e(idx);
    w = w(:,idx);
    b = b(idx);

    time_series_matrix(:,:,ii) = real(w*diag(b)*exp(e*t));

    if (any(b> 1000))
        e_total(:,ii) = nan+0i;
        b_total(:,ii) = nan;
        w_total(:,:,ii) = nan+0i;
    else
        e_total(:,ii) = e(idx);
        b_total(:,ii) = b(idx);
        w_total(:,:,ii) = w(:,idx);
    end
end


% REMOVE NAN 
e_total_no_nan = ones([(number_of_time_delays+1)*2,number_of_bagging])+1i;
b_total_no_nan = ones([(number_of_time_delays+1)*2,number_of_bagging]); 
w_total_no_nan = ones([(number_of_time_delays+1)*2,(number_of_time_delays+1)*2,number_of_bagging])+1i;
j=1;
for ii=1:number_of_bagging
    if (sum(isnan(b_total(:,ii))) == 0)
        e_total_no_nan(:,j) = e_total(:,ii);
        b_total_no_nan(:,j) = b_total(:,ii);
        w_total_no_nan(:,:,j) = w_total(:,:,ii);
        j=j+1;
    end
end

e_total = e_total_no_nan(:,1:j-1);
b_total = b_total_no_nan(:,1:j-1);
w_total = w_total_no_nan(:,:,1:j-1);


%% MEAN VALUE CALCULATION 

e_mean = mean(e_total, 2);
b_mean = mean(b_total, 2);
w_mean = mean(w_total, 3);

time_series_MeanValue = real(w_mean*diag(b_mean)*exp(e_mean*t));

%% PLOT DATA OF SH

subplot(4,1,4);
title('Predator-prey population prediction with Time Delay BOP-DMD','FontSize',15)
hold on
patch([1845 t_train(end)+1845 t_train(end)+1845 1845], [500 500 -500 -500], [0.7 0.7 0.7], EdgeColor = 'none')
hold off
alpha(0.3)
hold all
plot(t+1845,X(1,:),'--b',LineWidth=2)
plot(t+1845,time_series_MeanValue(1,:),'b', LineWidth=2);
plot(t+1845,X(2,:),'--r',LineWidth=2)
plot(t+1845,time_series_MeanValue(number_of_time_delays*2-1,:),'r', LineWidth=2);
hold off

legend('Training dataset','Snowshoe hare','Snowshoe hare prediction','Canadian lynx','Canadian lynx predicted','FontSize', 15)
xlabel('Time [years]', 'FontSize', 20)
ylabel('Number of animals', 'FontSize', 20)
ax = gca; 
ax.FontSize = 13; 
ylim([-10 180])
xlim([1845 1845+60])
grid on


