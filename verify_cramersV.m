function verify_cramersV()
% VERIFY_CRAMERSV 接受一个列联表，计算 Cramér's V，并验证与给定值是否一致
%
% 用法：
%   1. 在下方 W 中填入列联表
%   2. 在 C_expected 中填入论文中报告的 Cramér's V
%   3. 运行 verify_cramersV
%
% 输出：
%   - 正确公式计算出的 Cramér's V
%   - 与给定值的差异
%   - 诊断信息（P、N、chi2、df、p）

    clc; clear; close all;

    %% ====== 输入：列联表 ======
    % 示例：表1（4位作家）
    W = [13  1  1  1;
          0 15  0  1;
          0  0 16  0;
          0  2  1 13];

    %% ====== 输入：论文中报告的 Cramér's V ======
    C_expected = 0.86;   % 表1中报告的 C = 0.86

    %% ====== 计算 ======
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   Cramér''s V 验证脚本                                    ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    % 显示输入表
    fprintf('【输入列联表 W】\n');
    disp(W);

    % 计算正确公式下的 Cramér's V
    [chi2, p, df, P, N, C_correct] = computeCramersV_correct(W);

    % 计算错误公式下的 Cramér's V（用于对比）
    C_wrong = computeCramersV_wrong(W);

    %% ====== 输出 ======
    fprintf('\n【基本统计量】\n');
    fprintf('  总观测数 P        = %d\n', P);
    fprintf('  类别数 N          = %d\n', N);
    fprintf('  自由度 df         = %d\n', df);
    fprintf('  卡方值 chi2       = %.4f\n', chi2);
    fprintf('  p值               = %.4e\n', p);

    fprintf('\n【Cramér''s V 对比】\n');
    fprintf('  论文报告值        = %.4f\n', C_expected);
    fprintf('  正确公式计算值    = %.4f  (E = P * p_i. * p_.j)\n', C_correct);
    fprintf('  错误公式计算值    = %.4f  (E = N * p_i. * p_.j)\n', C_wrong);

    fprintf('\n【验证结果】\n');
    diff_correct = abs(C_correct - C_expected);
    diff_wrong   = abs(C_wrong   - C_expected);

    fprintf('  |C_correct - C_expected| = %.4f\n', diff_correct);
    fprintf('  |C_wrong   - C_expected| = %.4f\n', diff_wrong);

    if diff_correct < 0.01
        fprintf('\n  ✓ 论文报告值与【正确公式】一致\n');
    elseif diff_wrong < 0.01
        fprintf('\n  ⚠️  论文报告值与【错误公式】一致\n');
        fprintf('     说明论文使用的是 E = N * p_i. * p_.j\n');
        fprintf('     需要修正为 E = P * p_i. * p_.j\n');
    else
        fprintf('\n  ❌ 论文报告值与两种公式都不一致\n');
        fprintf('     可能原因：\n');
        fprintf('       1) 列联表数据有误\n');
        fprintf('       2) 论文报告值有误\n');
        fprintf('       3) 使用了其他计算方式\n');
    end

    %% ====== 可选：显示中间计算过程 ======
    fprintf('\n【中间计算过程】\n');
    rowTotals = sum(W, 2);
    colTotals = sum(W, 1);
    E_correct = (rowTotals * colTotals) / P;
    E_wrong   = (rowTotals * colTotals) / N;

    fprintf('  行和 R_i: '); fprintf('%d ', rowTotals); fprintf('\n');
    fprintf('  列和 C_j: '); fprintf('%d ', colTotals); fprintf('\n');

    fprintf('\n  期望频数 E（正确，用 P）:\n');
    disp(round(E_correct, 4));

    fprintf('  期望频数 E（错误，用 N）:\n');
    disp(round(E_wrong, 4));

    %% ====== 卡方分量对比 ======
    fprintf('\n【卡方分量对比】\n');
    chi2_components_correct = ((W - E_correct).^2) ./ E_correct;
    chi2_components_wrong   = ((W - E_wrong).^2) ./ E_wrong;

    fprintf('  正确公式的卡方分量:\n');
    disp(round(chi2_components_correct, 4));
    fprintf('  正确公式 chi2 = %.4f\n', sum(chi2_components_correct(:)));

    fprintf('\n  错误公式的卡方分量:\n');
    disp(round(chi2_components_wrong, 4));
    fprintf('  错误公式 chi2 = %.4f\n', sum(chi2_components_wrong(:)));

    fprintf('\n');
end

%% ========== 正确公式 ==========
function [chi2, p, df, P, N, C] = computeCramersV_correct(W)
% 正确公式：期望频数用 P（总观测数）
%
%   E_ij = P * p_i. * p_.j = (R_i * C_j) / P
%   chi2 = sum((O_ij - E_ij)^2 / E_ij)
%   C    = sqrt(chi2 / (P * (N - 1)))

    P = sum(W(:));
    [r, c] = size(W);
    N = min(r, c);

    if P == 0 || N <= 1
        chi2 = 0; p = 1; df = 0; C = 0;
        return;
    end

    rowTotals = sum(W, 2);
    colTotals = sum(W, 1);

    % 期望频数（正确：用 P）
    E = (rowTotals * colTotals) / P;
    E(E == 0) = eps;

    chi2 = sum(((W - E).^2) ./ E, 'all');
    df = (r - 1) * (c - 1);

    if df > 0
        p = 1 - chi2cdf(chi2, df);
    else
        p = 1;
    end

    C = sqrt(chi2 / (P * (N - 1)));
end

%% ========== 错误公式（用于对比） ==========
function C = computeCramersV_wrong(W)
% 错误公式：期望频数用 N（类别数）—— 论文修订版中的写法
%
%   E_ij = N * p_i. * p_.j
%   chi2 = sum((O_ij - E_ij)^2 / E_ij)
%   C    = sqrt(chi2 / (P * (N - 1)))

    P = sum(W(:));
    [r, c] = size(W);
    N = min(r, c);

    if P == 0 || N <= 1
        C = 0;
        return;
    end

    rowTotals = sum(W, 2);
    colTotals = sum(W, 1);

    % 边际比例
    p_i = rowTotals / P;
    p_j = colTotals / P;

    % 期望频数（错误：用 N）
    E = N * (p_i * p_j);
    E(E == 0) = eps;

    chi2 = sum(((W - E).^2) ./ E, 'all');
    C = sqrt(chi2 / (P * (N - 1)));
end
