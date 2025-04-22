% Lexuanthang.official@gmail.com,Lexuanthang.official@outlook.com
% Le Xuan Thang, 2023
% Tutorial: dynamic analysis: direct method: frequency domain
% Units: m, N

%% Bước 1: Nhập mô hình
CuaRaoBridge;
L=L_span;

% Nhập trường hợp hư hỏng 1 phần tử/ 1 lần

Matrix_Case = [0:10; % Thứ tự trường hợp / label
    0 1 2 3 4 5 6 82 100 91 14; % phần tử chịu hư hỏng
    0 10 20 30 40 50 10 20 10 30 10]; % Phần trăm hư hỏng


Materials0 = Materials;
Elements0 = Elements;

for i = 1:size(Matrix_Case, 2)
    Case = Matrix_Case(1, i);
    Element = Matrix_Case(2, i);
    Damage = Matrix_Case(3, i);
    Materials = Materials0;
    Elements = Elements0;
    % Update Materials and Elements based on the current case
    Materials = [Materials; 10 2E11*(1-Damage/100) 0.3 7800];
    Elements(Elements(:,1) == Element, 4) = 10;

    %check frequency
    % Assembly of stiffness matrix K
    [K,M]=asmkm(Nodes,Elements,Types,Sections,Materials,DOF);

    % Eigenvalue problem
    nMode=12;
    [~,omega]=eigfem(K,M,nMode);
    frequency = omega/2/pi;

%% BƯớc 2: Chọn loại phương tiện tác động lên cầu, chọn làn xe chạy, chọn tần số lấy mẫu
% Train/Vehicles phương tiện
    P = [ -10000 -15000 -15000; % P1 P2 P3 / Lực trục
    0 5 10]; % 0 l2 l3 khoảng cách giữa các trục

    DTBB =30; % [m] Distance train/truck to bridge before (front axle) khoảng cách giữa bánh xe và đầu cầu
    V = 150*1000/3600; % km/h --> [m/s] Velocity / Vận tốc chạy
    LT = sum(P(2,:)); % [m] Length of train/truck 

    seldof = reprow([600],1,8,[1])+0.03; %% *** Chọn làn mà xe chạy ***

    seldof = seldof(:);
    dt = 0.002; % Time step/resolution *** Chọn bước thời gian / tần số lấy mẫu***
%     t1 = DTBB/V; % [s] truck/train enter (firsr axle)
%     t2 = (DTBB + L + LT)/V; % [s] truck/train leave (last axle)
%     t3 = 2; % Thời gian bắt đầu ghi

    % Sampling parameters: time domain
    T = ((DTBB + L + LT)/V) + 100; % Time window [s]
    N = fix(T/dt); % Number of samples 10739
    t = (0:N)*dt; % Time axis (samples)

    % Sampling parameters: frequency domain
    F=1/dt; % Sampling frequency [Hz]
    df=1/T; % Frequency resolution
    f=(0:fix(N/2)-1)*df; % Positive frequencies corresponding to FFT [Hz]
    Omega=2*pi*f; % [rad/s] excitation frequency content

    % Excitation: transfer PLoad vector to nodalforce vector
    % (seldof --> all DOF)
%% Bước 3: Chọn các thông số lặp cho hàm gán lực vào kết cấu
    startInterval = 2; % Thời gian tàu bắt đầu vào cầu/ thời gian bắt đầu ghi dữ liệu 
    nloop = 8; % Số lần chạy của phương tiện trên cầu
    Pulse = -500; % Lực xung kích Pulse tác động lên cầu / Kết cấu
    gap = 10;  % Khoảng cách giữa các lần chạy của phương tiện trên cầu
    PLoad = trainload(P,L,DTBB,V,dt,seldof,Nodes,startInterval,Pulse,T, nloop, gap,f0); % [seldof x N samples]

%%
    % Eigenvalue analysis
    nMode = 12; % Number of modes to take into account
    [phi,omega]=eigfem(K,M,nMode); % Calculate eigenmodes and eigenfrequencies
    xi=0.07; % Constant modal damping ratio

    Pnodal = zeros(size(DOF,1),N);
    for itime = 1:N
        Pnodal(:,itime) = nodalvalues(DOF,seldof,PLoad(:,itime));
    end

    % Modal excitation
    Pm_ = phi.'*Pnodal; % [DOF,nMode] x [DOF, N samples]

    % Transfer nodal force vector time history to frequency domain
    Q = zeros(nMode,fix(N/2)); % keep positive frequency ONLY
    for inMode = 1:nMode
        temp = fft(Pm_(inMode,:));
        Q(inMode,:) = temp(1:fix(N/2));
    end

    Pm = Q;

    % Modal analysis: calc. the modal transfer functions and the modal disp.
    [X,H]=msupf(omega,xi,Omega,Pm); % Modal response, positive freq (nMode * N/2)
    % F-dom -> t-dom [inverse Fourier transform]
    X = [X, zeros(nMode,1), conj(X(:,end:-1:2))];
    x = ifft(X,[],2); % Modal response (nMode * N)

    % Modal displacements -> nodal displacements
    u=phi*x; % Nodal response (nDOF * N)
    
%% Bước 4: Chọn sensors/ điểm nodes xuất kết quả
    d = [reprow([101,301],1,6,[1,1])+0.03;reprow([101,301],1,6,[1,1])+0.02];
    c = selectdof(DOF,d(:));
    u_c = c*u;

    acceleration = zeros(size(u_c));
    for i_acc = 1:size(u_c,1)
        acceleration(i_acc,:) = displacementToAcceleration(u_c(i_acc,:), dt);
        figure;
        plot(acceleration(i_acc,:));
        title(["acc" i_acc]);
        xlabel("samples");
        ylabel("acc");
    end

    filename = sprintf('D:/onedrive/Detai/Code/TrainLoadCode/Data/CuaRao%d.mat', Case);
    save(filename, 'acceleration');
end
