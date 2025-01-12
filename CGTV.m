%% Parameter settings
lambda_tv = 0.01; % Fixed TV regularization parameter
epsilon = 1e-6; % Relaxation term for gradient calculation
max_iter = 1000; % Maximum number of iterations
tolerance = 1e-10; % Convergence tolerance

% Initialize variables
x = zeros(size(JJ, 2), 1); % Initial solution (4096x1)
r = JJ' * (b_256 - JJ * x); % Initial residual
d = r; % Initial search direction
prev_r_norm = norm(r); % Initial residual norm

%% CGTV algorithm iteration
for k = 1:max_iter
    % Compute TV gradient
    g_tv = compute_tv_grad_vectorized(x, epsilon); % Vectorized TV gradient

    % Compute total gradient
    grad = JJ' * (JJ * d) + lambda_tv * g_tv;

    % Dynamically adjust the step size
    alpha = (r' * r) / (d' * grad);
    alpha = max(alpha, 1e-9); % Prevent step size from being too small
    alpha = min(alpha, 2e-2); % Limit the maximum step size

    % Update variables
    x = x + alpha * d;
    r_new = r - alpha * (JJ' * (JJ * d));

    % Convergence check
    r_norm = norm(r_new);
    if abs(r_norm - prev_r_norm) / prev_r_norm < tolerance
        fprintf('Converged at iteration %d\n', k);
        final_iteration = k; % Save the final number of iterations
        break;
    end
    prev_r_norm = r_norm;

    % Update search direction
    beta = (r_new' * r_new) / (r' * r);
    d = r_new + beta * d;
    r = r_new;
end

% Restore image dimensions
reconstructed_image = reshape(x, 64, 64);

% Set negative values to zero
reconstructed_image(reconstructed_image < 0) = 0;

%% Vectorized TV gradient computation function
function g_tv = compute_tv_grad_vectorized(x, epsilon)
    img = reshape(x, 64, 64); % Convert vector x to an image matrix
    % Compute gradients in the horizontal and vertical directions
    dx = diff([img; img(end, :)], 1, 1); % Gradient in the x direction
    dy = diff([img, img(:, end)], 1, 2); % Gradient in the y direction
    grad = sqrt(dx.^2 + dy.^2 + epsilon); % Smoothed TV gradient
    g_tv = grad(:); % Return as a column vector
end