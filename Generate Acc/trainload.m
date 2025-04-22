function [PLoad]=trainload(P, L, DTBB, V, dt, seldof, Nodes, firstime, Pulse, T, nloop,gap,f0)
    %TRAINLOAD   Calculate force going through the bridge.
    
    LT = sum(P(2,:)); % Length of train/trucks
    entryTimes = zeros(1, nloop);
    leaveTimes = zeros(1, nloop);
    
    % Initialize entry and leave times for each train crossing
    for lo = 1:nloop
        if lo == 1
            entryTimes(lo) = DTBB/V + firstime;
            leaveTimes(lo) = (DTBB + L + LT)/V + firstime;
        else
            entryTimes(lo) = DTBB/V + firstime + leaveTimes(lo-1) + gap; % +10 for the gap
            leaveTimes(lo) = (DTBB + L + LT)/V + firstime + leaveTimes(lo-1) + gap;
        end
    end

    % Time vector
    N = ceil(T/dt);
    t = (0:N-1)*dt;

    % Nodal loads
    PLoad = zeros(size(seldof, 1), N);
    sigma = 0.1; % Standard deviation for noise

    % Calculate loads
    for itime = 1:N
        periodic_force = sum(sin(2 * pi * f0 * t(itime)));
        for lo = 1:nloop
            % Check if within time range of train crossing
            if t(itime) >= entryTimes(lo) && t(itime) <= leaveTimes(lo)
                for iload = 1:size(P, 2)
                    dis = V*(t(itime) - entryTimes(lo)) - sum(P(2, 1:iload));
                    for ielem = 1:(size(seldof, 1)-1)
                        x_1 = Nodes(Nodes(:, 1) == fix(seldof(ielem)), 2);
                        x_2 = Nodes(Nodes(:, 1) == fix(seldof(ielem+1)), 2);
                        if dis >= x_1 && dis <= x_2
                            PLoad(ielem+1, itime) = PLoad(ielem+1, itime) + (P(1, iload)*(dis-x_1)/(x_2-x_1))+periodic_force;
                            PLoad(ielem, itime) = PLoad(ielem, itime) + (P(1, iload)*(x_2-dis)/(x_2-x_1))+periodic_force;
                        end
                    end
                end
            end
        end

        % Check if the train has just left
        if any(t(itime) > leaveTimes)

            % Apply this periodic force to a random position on the bridge
            random_position = rand * L;
            for ielem = 1:(size(seldof, 1)-1)
                x_1 = Nodes(Nodes(:, 1) == fix(seldof(ielem)), 2);
                x_2 = Nodes(Nodes(:, 1) == fix(seldof(ielem+1)), 2);
                if random_position >= x_1 && random_position <= x_2
                    PLoad(ielem+1, itime) = PLoad(ielem+1, itime) + (periodic_force * (random_position-x_1)/(x_2-x_1));
                    PLoad(ielem, itime) = PLoad(ielem, itime) + (periodic_force * (x_2-random_position)/(x_2-x_1));
                end
            end
        end

    % Thêm các yếu tố môi trường và giao thông địa phương
    wind_effect = 0.05; % Thay đổi để phù hợp với ảnh hưởng của gió
    local_traffic_effect = 0.02; % Ảnh hưởng từ giao thông địa phương

        if all(t(itime) < entryTimes | t(itime) > leaveTimes)
            noise = sigma * randn(size(PLoad, 1), 1);
            PLoad(:,itime) = PLoad(:,itime) + periodic_force*noise;
            PLoad(:,itime) = PLoad(:,itime) + Pulse*noise;
           % Thêm ảnh hưởng từ gió và giao thông địa phương
            PLoad(:, itime) = PLoad(:, itime) + wind_effect * noise;
            PLoad(:, itime) = PLoad(:, itime) + local_traffic_effect * noise;
        end
    end
end
