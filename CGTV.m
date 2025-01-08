%% ��������
lambda_tv = 0.01; % �̶� TV ���򻯲���
epsilon = 1e-6; % �ݶȼ�����ɳ���
max_iter = 1000; % ����������
tolerance = 1e-10; % ��������

% ��ʼ������
x = zeros(size(JJ, 2), 1); % ��ʼ�� (4096x1)
r = JJ' * (b_256 - JJ * x); % ��ʼ�в�
d = r; % ��ʼ��������
prev_r_norm = norm(r); % ��ʼ�в��

%% CGTV�㷨����
for k = 1:max_iter
    % ���� TV �ݶ�
    g_tv = compute_tv_grad_vectorized(x, epsilon); % ������ TV �ݶ�

    % �������ݶ�
    grad = JJ' * (JJ * d) + lambda_tv * g_tv;

    % ��̬��������
    alpha = (r' * r) / (d' * grad);
    alpha = max(alpha, 1e-9); % ��ֹ������С
    alpha = min(alpha, 2e-2); % ���Ʋ������ֵ

    % ���±���
    x = x + alpha * d;
    r_new = r - alpha * (JJ' * (JJ * d));

    % �����ж�
    r_norm = norm(r_new);
    if abs(r_norm - prev_r_norm) / prev_r_norm < tolerance
        fprintf('Converged at iteration %d\n', k);
        final_iteration = k; % �������յ�������
        break;
    end
    prev_r_norm = r_norm;

    % ������������
    beta = (r_new' * r_new) / (r' * r);
    d = r_new + beta * d;
    r = r_new;
end

% �ָ�ͼ��ߴ�
reconstructed_image = reshape(x, 64, 64);

% ����ֵ����
reconstructed_image(reconstructed_image < 0) = 0;

%% ������ TV �ݶȼ��㺯��
function g_tv = compute_tv_grad_vectorized(x, epsilon)
    img = reshape(x, 64, 64); % ������ x ת��Ϊͼ�����
    % ����ˮƽ�ʹ�ֱ�����ݶ�
    dx = diff([img; img(end, :)], 1, 1); % x �����ݶ�
    dy = diff([img, img(:, end)], 1, 2); % y �����ݶ�
    grad = sqrt(dx.^2 + dy.^2 + epsilon); % ƽ���� TV �ݶ�
    g_tv = grad(:); % չ��Ϊ����������
end