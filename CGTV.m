%% 参数设置
lambda_tv = 0.01; % 固定 TV 正则化参数
epsilon = 1e-6; % 梯度计算的松弛项
max_iter = 1000; % 最大迭代次数
tolerance = 1e-10; % 收敛精度

% 初始化变量
x = zeros(size(JJ, 2), 1); % 初始解 (4096x1)
r = JJ' * (b_256 - JJ * x); % 初始残差
d = r; % 初始搜索方向
prev_r_norm = norm(r); % 初始残差范数

%% CGTV算法迭代
for k = 1:max_iter
    % 计算 TV 梯度
    g_tv = compute_tv_grad_vectorized(x, epsilon); % 向量化 TV 梯度

    % 计算总梯度
    grad = JJ' * (JJ * d) + lambda_tv * g_tv;

    % 动态调整步长
    alpha = (r' * r) / (d' * grad);
    alpha = max(alpha, 1e-9); % 防止步长过小
    alpha = min(alpha, 2e-2); % 限制步长最大值

    % 更新变量
    x = x + alpha * d;
    r_new = r - alpha * (JJ' * (JJ * d));

    % 收敛判断
    r_norm = norm(r_new);
    if abs(r_norm - prev_r_norm) / prev_r_norm < tolerance
        fprintf('Converged at iteration %d\n', k);
        final_iteration = k; % 保存最终迭代步数
        break;
    end
    prev_r_norm = r_norm;

    % 更新搜索方向
    beta = (r_new' * r_new) / (r' * r);
    d = r_new + beta * d;
    r = r_new;
end

% 恢复图像尺寸
reconstructed_image = reshape(x, 64, 64);

% 将负值置零
reconstructed_image(reconstructed_image < 0) = 0;

%% 向量化 TV 梯度计算函数
function g_tv = compute_tv_grad_vectorized(x, epsilon)
    img = reshape(x, 64, 64); % 将向量 x 转换为图像矩阵
    % 计算水平和垂直方向梯度
    dx = diff([img; img(end, :)], 1, 1); % x 方向梯度
    dy = diff([img, img(:, end)], 1, 2); % y 方向梯度
    grad = sqrt(dx.^2 + dy.^2 + epsilon); % 平滑的 TV 梯度
    g_tv = grad(:); % 展开为列向量返回
end