function acceleration = displacementToAcceleration(displacement, dt)
    % displacement: Array of displacement values
    % dt: Time step between measurements

    % First derivative (Velocity)
    velocity = diff(displacement) / dt;

    % Second derivative (Acceleration)
    % The 'diff' function reduces the size of the array by 1, so apply diff again on velocity and divide by dt
    acceleration = diff(velocity) / dt;

    % To maintain the original size, you can pad the acceleration array with zeros or use other methods like interpolation
    acceleration = [0, acceleration, 0]; % Padding with zeros as an example
end
