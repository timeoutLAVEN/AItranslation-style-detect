function main()
    % 主函数 - 基于压缩的作者识别系统（全局枚举版）
    % 只保留：顺序截取 + LZMA/PPMd（7-Zip）
    % 方案B：失败切片跳过并记录，不使用随机回退
    % 无条件固定随机种子，代码层面无随机源
    %
    % 本版本额外实现：
    %   方案2 - Cramér's V 计算前剔除零行零列（统计修正）
    %   方案4 - 零预测成因诊断（error analysis）
    %   方案A - 预计算训练文本压缩大小，避免重复 7z 调用（性能优化）
    %   修复 - 零预测流向打印全部非零类别（不再截断到前3）
    %   ★ 全局枚举 - worst2 (C(n,2))、top4 (C(n,4))、top3 (C(n,3)) 全部枚举
    %   ★ 样本量保护 - n < minSlices 的组合排除
    %   ★ 候选排名表 - 论文附录可直接引用

    try
        cleanupTempFolder();
        rng(2023, 'twister');

        % ===== 路径配置 =====
        trainFolder = 'C:\Users\han\Desktop\distiguisha';
        testFolder  = 'C:\Users\han\Desktop\distiguishb';

        % ===== 配置参数 =====
        config = getConfiguration();

        % ===== 阶段1：准备训练数据 =====
        fprintf('\n=== 阶段1：准备训练数据 ===\n');
        trainingData = prepareTrainingData(trainFolder, config);

        % ===== 阶段2：准备测试数据 =====
        fprintf('\n=== 阶段2：准备测试数据 ===\n');
        testData = prepareTestData(testFolder, config, trainingData);

        % ===== 阶段3：全局枚举所有 2/3/4 人组合 =====
        fprintf('\n=== 阶段3：全局枚举所有组合（2人/3人/4人）===\n');
        subsetResults = enumerateAllSubsets(trainingData, testData, config);

        worst2_authors = subsetResults.best2.authors;
        worst2_V       = subsetResults.best2.V;
        fprintf('\n✓ worst2 (V最大，风格差异最大): %s + %s, V = %.4f\n', ...
                worst2_authors{1}, worst2_authors{2}, worst2_V);

        top4_authors = subsetResults.best4.authors;
        top4_V       = subsetResults.best4.V;
        fprintf('✓ top4 (V最小，全局最优): %s, V = %.4f\n', ...
                strjoin(top4_authors, ' + '), top4_V);

        top3_authors = subsetResults.best3.authors;
        top3_V       = subsetResults.best3.V;
        fprintf('✓ top3 (V最小，全局最优): %s, V = %.4f\n', ...
                strjoin(top3_authors, ' + '), top3_V);

        % ===== 阶段4：完整 N 人分类 =====
        fprintf('\n=== 阶段4：执行完整分类 ===\n');
        [resultTable, failStats] = performClassification(trainingData, testData, config.algorithm);

        % ===== 阶段4.5：诊断 =====
        fprintf('\n=== 阶段4.5：诊断分析 ===\n');
        diagnoseAbnormalCramersV(trainingData, testData, config.algorithm, 'CTR (中→俄)');

        % ===== 阶段5：保存 =====
        fprintf('\n=== 阶段5：保存结果 ===\n');
        authors = {trainingData.author};
        saveResults(resultTable, trainFolder, config, authors, failStats, subsetResults);

        % ===== 汇总 =====
        fprintf('\n');
        fprintf('╔════════════════════════════════════════════╗\n');
        fprintf('║            分析结果汇总                    ║\n');
        fprintf('║  V小 = 风格可区分性低 = 风格一致性高       ║\n');
        fprintf('║  V大 = 风格可区分性高 = 风格差异大         ║\n');
        fprintf('║  注：不同 k 下的 V 不可直接横向比较         ║\n');
        fprintf('╚════════════════════════════════════════════╝\n');

        six_V = calculateCramersV(resultTable);
        fprintf('6人全部 (k=6)                 V = %.4f, n = %d\n', ...
                six_V, sum(resultTable(:)));
        fprintf('worst2  (k=2): %-20s V = %.4f, n = %d\n', ...
                strjoin(worst2_authors, '+'), worst2_V, subsetResults.best2.n);
        fprintf('top4    (k=4): %-20s V = %.4f, n = %d\n', ...
                strjoin(top4_authors, '+'), top4_V, subsetResults.best4.n);
        fprintf('top3    (k=3): %-20s V = %.4f, n = %d\n', ...
                strjoin(top3_authors, '+'), top3_V, subsetResults.best3.n);

        fprintf('\n【论文用提示】\n');
        fprintf('  - 不同 k 的 V 来自不同列联表，不可直接比较大小\n');
        fprintf('  - 若需横向比较，请固定每个类别的切片数（固定 n）\n');
        fprintf('  - V 仅代表"风格可区分性"，不等价于绝对翻译质量\n');
        fprintf('  - 建议结合 BLEU/COMET/人工评分做外部效度验证\n');
        fprintf('=== 分析完成! ===\n');

    catch ME
        fprintf('\n=== 程序执行失败 ===\n');
        fprintf('错误信息: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
        end
        cleanupTempFolder();
        rethrow(ME);
    end
end

%% ========== 配置管理 ==========
function config = getConfiguration()
    config.trainSizeKB = validateInput('请输入训练集大小(KB)：', 10, 1000);
    config.testSliceSizeKB = validateInput('请输入测试切片大小(KB)：', 1, config.trainSizeKB);

    config.testSliceCount = floor(config.trainSizeKB / config.testSliceSizeKB);
    if config.testSliceCount < 1
        error('错误：切片大小不能大于或等于训练集大小！');
    end

    config.isRandom = false;
    fprintf('\n截取方式：顺序截取（精简版固定）\n');

    config.algorithm = selectCompressionAlgorithm();
    config.requiredBytes = (config.trainSizeKB + config.testSliceCount * config.testSliceSizeKB) * 1024;

    fprintf('自动计算：每个作者使用 %d 个 %dKB 测试切片（每个文件需 ≥ %dKB）\n', ...
            config.testSliceCount, config.testSliceSizeKB, config.requiredBytes/1024);

    % ===== 全局枚举相关参数 =====
    % minSlices: 组合进入排序前的最小有效切片数；避免小样本噪声主导
    % 默认 30。若切片总数较少，可下调为 testSliceCount*3 左右
    config.minSlices = 30;

    % fixedSlicesPerAuthor: 论文横向比较用，固定每作者切片数
    % 若不需要固定 n，设为 Inf
    config.fixedSlicesPerAuthor = Inf;

    verifyCompressionTools(config.algorithm);
end

function value = validateInput(prompt, minVal, maxVal)
    while true
        value = input(prompt);
        if isnumeric(value) && isscalar(value) && ...
           value == floor(value) && value >= minVal && value <= maxVal
            return;
        end
        fprintf('输入必须为[%d-%d]之间的整数\n', minVal, maxVal);
    end
end

function compressionAlgorithm = selectCompressionAlgorithm()
    fprintf('\n可用压缩算法:\n');
    fprintf('1) LZMA (7-Zip)\n');
    fprintf('2) PPMd (7-Zip)\n');
    while true
        compressionAlgorithm = input('请选择压缩算法 (1-2)：');
        if compressionAlgorithm == 1
            compressionAlgorithm = 2;
            return;
        elseif compressionAlgorithm == 2
            compressionAlgorithm = 6;
            return;
        end
        fprintf('输入错误，请输入1或2\n');
    end
end

%% ========== 全局枚举所有 2/3/4 人组合 ==========
function subsetResults = enumerateAllSubsets(trainingData, testData, config)
    % 全局枚举所有 2/3/4 人组合，返回每个规模的完整排名与最优组合
    %
    % 输出结构：
    %   .allK      : struct array, 每个元素含 S, authors, V, n, k, nZeroRow, nZeroCol, table
    %   .rankK     : 按 V 升序排序后的（仅有效样本）
    %   .best2     : V 最大（worst2）
    %   .best3     : V 最小（top3）
    %   .best4     : V 最小（top4）

    % ---- 论文模式：固定每个作者的切片数 ----
    if isfinite(config.fixedSlicesPerAuthor)
        for i = 1:numel(testData)
            if numel(testData(i).slices) > config.fixedSlicesPerAuthor
                testData(i).slices = testData(i).slices(1:config.fixedSlicesPerAuthor);
            end
        end
        fprintf('[固定n模式] 每作者最多取 %d 个切片\n', config.fixedSlicesPerAuthor);
    end

    authors_all = {trainingData.author};
    nTotal = numel(authors_all);
    minSlices = config.minSlices;

    subsetResults = struct();

    for k = 2:4
        subsets = nchoosek(1:nTotal, k);
        nComb = size(subsets, 1);

        fprintf('\n--- 枚举 %d 人组合: 共 %d 种 ---\n', k, nComb);
        fprintf('%-3s | %-40s | %8s | %6s | %4s | %8s\n', ...
                'No.', '组合', 'V', 'n', 'k', '退化');
        fprintf('%s\n', repmat('-', 1, 80));

        rec = struct('S', {}, 'authors', {}, 'V', {}, 'n', {}, 'k', {}, ...
                     'nZeroRow', {}, 'nZeroCol', {}, 'table', {});

        for c = 1:nComb
            S = subsets(c, :);
            trainingData_sub = trainingData(S);
            testData_sub = testData(ismember({testData.author}, authors_all(S)));

            [resultTable_sub, ~] = performClassification( ...
                trainingData_sub, testData_sub, config.algorithm);

            rowT = sum(resultTable_sub, 2);
            colT = sum(resultTable_sub, 1);
            nZeroRow = sum(rowT == 0);
            nZeroCol = sum(colT == 0);
            tbl_eff = resultTable_sub(rowT > 0, colT > 0);

            nEff = sum(tbl_eff(:));
            kEff = min(size(tbl_eff));
            V = calculateCramersV(resultTable_sub);

            rec(end+1).S = S;                              %#ok<AGROW>
            rec(end).authors = authors_all(S);
            rec(end).V = V;
            rec(end).n = nEff;
            rec(end).k = kEff;
            rec(end).nZeroRow = nZeroRow;
            rec(end).nZeroCol = nZeroCol;
            rec(end).table = resultTable_sub;

            flag = '';
            if nEff < minSlices
                flag = '⚠低样本';
            end
            fprintf('%-3d | %-40s | %8.4f | %6d | %4d | 行%d 列%d %s\n', ...
                    c, strjoin(authors_all(S), '+'), V, nEff, kEff, ...
                    nZeroRow, nZeroCol, flag);
        end

        % ---- 剔除低样本组合 ----
        validMask = [rec.n] >= minSlices;
        recValid = rec(validMask);
        if isempty(recValid)
            warning('%d 人组合中没有任何组合满足 n >= %d，回退到全部组合', ...
                    k, minSlices);
            recValid = rec;
        end

        % V 升序
        [~, orderAsc] = sort([recValid.V], 'ascend');
        recValidAsc = recValid(orderAsc);

        subsetResults.(sprintf('all%d', k)) = rec;
        subsetResults.(sprintf('rank%d', k)) = recValidAsc;

        if k == 2
            [~, orderDesc] = sort([recValid.V], 'descend');
            subsetResults.best2 = recValid(orderDesc(1));
        else
            subsetResults.(sprintf('best%d', k)) = recValidAsc(1);
        end
    end

    % ---- 汇总打印 ----
    fprintf('\n╔════════════════════════════════════════════╗\n');
    fprintf('║  全局枚举结果（已剔除 n < %d 的组合）      ║\n', minSlices);
    fprintf('╚════════════════════════════════════════════╝\n');

    printTopK(subsetResults.rank2, 'worst2 (V最大)', 5, 'descend');
    printTopK(subsetResults.rank4, 'top4  (V最小)', 5, 'ascend');
    printTopK(subsetResults.rank3, 'top3  (V最小)', 5, 'ascend');
end

function printTopK(recList, titleStr, K, direction)
    fprintf('\n【%s】前 %d 名:\n', titleStr, min(K, numel(recList)));
    if isempty(recList)
        fprintf('  (无有效组合)\n');
        return;
    end

    if strcmp(direction, 'descend')
        [~, ord] = sort([recList.V], 'descend');
    else
        [~, ord] = sort([recList.V], 'ascend');
    end

    fprintf('  %-3s | %-35s | %8s | %6s\n', 'No.', '组合', 'V', 'n');
    for i = 1:min(K, numel(ord))
        r = recList(ord(i));
        fprintf('  %-3d | %-35s | %8.4f | %6d\n', ...
                i, strjoin(r.authors, '+'), r.V, r.n);
    end
end

%% ========== 异常诊断 ==========
function diagnoseAbnormalCramersV(trainingData, testData, algorithm, languagePair)
    fprintf('\n╔════════════════════════════════════════════╗\n');
    fprintf('║  诊断异常 Cramér V 值: %s\n', languagePair);
    fprintf('╚════════════════════════════════════════════╝\n');

    [resultTable, failStats] = performClassification(trainingData, testData, algorithm);
    cramerV = calculateCramersV(resultTable);

    authors = {trainingData.author};
    n_authors = length(authors);

    fprintf('\n【基础统计】\n');
    fprintf('  Cramér V 值（剔除退化类别后）: %.4f\n', cramerV);
    fprintf('  有效切片数: %d\n', sum(resultTable(:)));
    fprintf('  总切片数: %d\n', failStats.totalSlices);
    fprintf('  失败切片数: %d\n', failStats.failedSlices);
    fprintf('  失败率: %.2f%%\n', failStats.failedRate*100);

    if sum(resultTable(:)) == 0
        fprintf('  ⚠️  无有效分类结果，诊断终止\n');
        return;
    end

    fprintf('  准确率: %.2f%%\n', sum(diag(resultTable))/sum(resultTable(:))*100);

    if failStats.failedRate > 0.05
        fprintf('\n  ⚠️  失败率 > 5%%，结果可靠性下降！\n');
    end

    fprintf('\n【混淆矩阵分析】\n');
    [chi2, p, df] = computeChiSquareFromTable(resultTable);
    fprintf('  卡方值 (χ²): %.4f  [基于剔除零行列后的表]\n', chi2);
    fprintf('  自由度: %d\n', df);
    fprintf('  P 值: %.4e\n', p);
    if p > 0.05
        fprintf('  ⚠️  P 值 > 0.05：两个变量独立（无显著关联）\n');
    else
        fprintf('  ✓ P 值 < 0.05：两个变量有显著关联\n');
    end

    fprintf('\n【行（真实标签）的分布】\n');
    rowTotals = sum(resultTable, 2);
    for i = 1:n_authors
        if sum(rowTotals) > 0
            fprintf('  %s: %d (%.1f%%)\n', authors{i}, rowTotals(i), rowTotals(i)/sum(rowTotals)*100);
        else
            fprintf('  %s: 0 (无有效切片)\n', authors{i});
        end
    end

    fprintf('\n【列（预测标签）的分布】\n');
    colTotals = sum(resultTable, 1);
    for j = 1:n_authors
        if sum(colTotals) > 0
            fprintf('  %s: %d (%.1f%%)\n', authors{j}, colTotals(j), colTotals(j)/sum(colTotals)*100);
        else
            fprintf('  %s: 0 (从未被预测)\n', authors{j});
        end
    end

    fprintf('\n【对角线分析（正确预测）】\n');
    diag_values = diag(resultTable);
    for i = 1:n_authors
        if rowTotals(i) > 0
            correct_rate = diag_values(i) / rowTotals(i) * 100;
            fprintf('  %s: %d/%d (%.1f%%)\n', authors{i}, diag_values(i), rowTotals(i), correct_rate);
        else
            fprintf('  %s: 0/0 (无有效切片)\n', authors{i});
        end
    end

    fprintf('\n【均匀分布检验】\n');
    expected_per_cell = sum(resultTable(:)) / (n_authors * n_authors);
    fprintf('  如果完全随机，每个格子应该约有 %.0f 个样本\n', expected_per_cell);
    fprintf('  实际混淆矩阵:\n');

    fprintf('     ');
    for j = 1:n_authors
        fprintf('%10s', authors{j});
    end
    fprintf('\n');
    for i = 1:n_authors
        fprintf('%8s:', authors{i});
        for j = 1:n_authors
            fprintf('%10d', resultTable(i,j));
        end
        fprintf('\n');
    end

    variance_from_uniform = var(resultTable(:));
    fprintf('\n  矩阵中所有值的方差: %.2f\n', variance_from_uniform);
    if variance_from_uniform < 10
        fprintf('  ⚠️  矩阵值过于均匀！可能是完全随机分类\n');
    else
        fprintf('  ✓ 矩阵分布合理\n');
    end

    fprintf('\n【预测偏向分析】\n');
    for j = 1:n_authors
        [most_predicted_count, most_predicted_for] = max(resultTable(:, j));

        if isempty(most_predicted_for) || most_predicted_count == 0 ...
                || most_predicted_for < 1 || most_predicted_for > n_authors
            fprintf('  预测为 %s: 无有效预测（整列为 0）\n', authors{j});
            continue;
        end

        tiedRows = find(resultTable(:, j) == most_predicted_count);
        if numel(tiedRows) > 1
            fprintf('  预测为 %s: 最多来自真实 %s (%d 个) [并列: %s]\n', ...
                    authors{j}, authors{most_predicted_for}, ...
                    most_predicted_count, strjoin(authors(tiedRows), ', '));
        else
            fprintf('  预测为 %s: 最多来自真实 %s (%d 个)\n', ...
                    authors{j}, authors{most_predicted_for}, most_predicted_count);
        end
    end

    zeroCols = find(colTotals == 0);
    if ~isempty(zeroCols)
        fprintf('\n【零预测成因诊断（方案4: error analysis）】\n');
        for k = 1:numel(zeroCols)
            j = zeroCols(k);
            diagnoseZeroPrediction(trainingData, testData, algorithm, j, authors, resultTable);
        end
    end

    fprintf('\n【压缩差异分析】\n');
    analyzeCompressionDifferences(trainingData, testData, algorithm);

    fprintf('\n【对标分析和诊断结论】\n');
    fprintf('  当前 V 值（剔除退化类别后）: %.4f\n', cramerV);
    fprintf('  CTE (中→英): 0.4-0.7  (参考值)\n');
    fprintf('  ETC (英→中): 0.3-0.7  (参考值)\n');
    fprintf('  CTR (中→俄): 本次结果\n');

    if cramerV < 0.15
        fprintf('\n  ❌ 异常判断：V 值远小于其他语言对\n');
        fprintf('  可能原因:\n');
        fprintf('    1) 分类算法在俄语上完全失效 - 预测几乎随机\n');
        fprintf('    2) 俄语文本的压缩特性相近 - 难以区分不同模型\n');
        fprintf('    3) 样本量过少 - 统计不足\n');
        fprintf('    4) 文本编码/解析问题 - 数据质量差\n');
        fprintf('    5) 俄语语言特性 - 字符集有限导致熵值低\n');
    elseif cramerV < 0.25
        fprintf('\n  ⚠️  偏低判断：V 值低于其他语言对\n');
        fprintf('  可能原因:\n');
        fprintf('    1) 俄语翻译风格相似度高 - 模型间区别不大\n');
        fprintf('    2) 压缩算法对俄语特性识别不足\n');
        fprintf('    3) 训练集样本特性导致的正常现象\n');
    else
        fprintf('\n  ✓ 正常范围\n');
    end
end

%% ========== 方案4：零预测成因诊断 ==========
function diagnoseZeroPrediction(trainingData, testData, algorithm, zeroIdx, authors, resultTable)
    targetAuthor = authors{zeroIdx};
    fprintf('\n  ── 零预测类别: %s ──\n', targetAuthor);

    targetTrain = [];
    for i = 1:length(trainingData)
        if strcmp(strtrim(trainingData(i).author), strtrim(targetAuthor))
            targetTrain = trainingData(i);
            break;
        end
    end

    if isempty(targetTrain)
        fprintf('    ⚠️  未找到该类别训练数据\n');
        return;
    end

    txt = targetTrain.text;
    ent = computeEntropy(txt);
    fprintf('    [4a] 训练文本特征:\n');
    fprintf('         长度: %d 字节 (%.2f KB)\n', length(txt), length(txt)/1024);
    fprintf('         字符熵: %.4f bits/char\n', ent);

    fprintf('         与其他类别对比:\n');
    for i = 1:length(trainingData)
        otherTxt = trainingData(i).text;
        otherEnt = computeEntropy(otherTxt);
        marker = '';
        if strcmp(strtrim(trainingData(i).author), strtrim(targetAuthor))
            marker = ' ← 目标';
        end
        fprintf('           %-12s 长度=%6d, 熵=%.4f%s\n', ...
                trainingData(i).author, length(otherTxt), otherEnt, marker);
    end

    fprintf('    [4b] 压缩距离排名分析:\n');
    allRanks = [];

    for t = 1:length(testData)
        if isempty(testData(t).slices), continue; end
        ts = testData(t).slices{1};
        diffs = zeros(1, length(trainingData));
        for i = 1:length(trainingData)
            diffs(i) = calculateCompressionDiff(trainingData(i).text, ts, algorithm, ...
                                                trainingData(i).trainCompressedSize);
        end
        [~, sortIdx] = sort(diffs, 'ascend');
        rankOfTarget = find(sortIdx == zeroIdx, 1);
        allRanks(end+1) = rankOfTarget; %#ok<AGROW>
    end

    if ~isempty(allRanks)
        fprintf('         在 %d 个测试切片中，%s 的压缩距离排名:\n', ...
                length(allRanks), targetAuthor);
        fprintf('           平均排名: %.2f / %d\n', mean(allRanks), length(trainingData));
        fprintf('           最小排名: %d (最好情况)\n', min(allRanks));
        fprintf('           最大排名: %d (最差情况)\n', max(allRanks));
        fprintf('           排名为 1 的次数: %d\n', sum(allRanks == 1));

        if mean(allRanks) <= 2
            fprintf('         诊断: ⚠️  %s 经常排第 2 名 → 分类器区分度不足（"差一点就赢"）\n', targetAuthor);
        elseif mean(allRanks) >= length(trainingData) - 1
            fprintf('         诊断: ❌ %s 的压缩距离总是最大 → 其特征与其他类别显著不同\n', targetAuthor);
        else
            fprintf('         诊断: %s 的压缩距离排名居中 → 被其他类别稳定压制\n', targetAuthor);
        end
    end

    fprintf('    [4c] 单向退化检查:\n');
    rowTotal = sum(resultTable(zeroIdx, :));
    colTotal = sum(resultTable(:, zeroIdx));
    fprintf('         真实 %s 的样本数 (行和): %d\n', targetAuthor, rowTotal);
    fprintf('         被预测为 %s 的样本数 (列和): %d\n', targetAuthor, colTotal);

    if rowTotal > 0 && colTotal == 0
        fprintf('         诊断: 单向退化 —— 该类别样本被误判为其他类，但其他类从不被误判为该类\n');

        fprintf('         该类别样本流向（全部非零，按降序）:\n');
        flows = resultTable(zeroIdx, :);
        [sortedFlow, sortIdx] = sort(flows, 'descend');
        nNonZero = 0;
        for k = 1:length(sortIdx)
            if sortedFlow(k) > 0
                nNonZero = nNonZero + 1;
                fprintf('           → %-12s : %3d (%.1f%%)\n', ...
                        authors{sortIdx(k)}, sortedFlow(k), ...
                        sortedFlow(k) / rowTotal * 100);
            end
        end
        fprintf('         流向类别数: %d / %d，总计 %d 个切片\n', ...
                nNonZero, length(authors), rowTotal);

        top1Share = sortedFlow(1) / rowTotal;
        pFlow = flows(flows > 0) / rowTotal;
        flowEntropyNorm = -sum(pFlow .* log2(pFlow)) / log2(length(authors));
        fprintf('         最大流向占比: %.1f%%（%s）\n', ...
                top1Share*100, authors{sortIdx(1)});
        fprintf('         流向分布归一化熵: %.4f（0=集中，1=均匀）\n', flowEntropyNorm);

    elseif rowTotal == 0
        fprintf('         诊断: 该类别无测试样本（行也为 0）\n');
    end
end

function ent = computeEntropy(txt)
    if isempty(txt)
        ent = 0;
        return;
    end
    bytes = double(txt(:));
    if ~isempty(bytes)
        counts = accumarray(bytes - min(bytes) + 1, 1);
        p = counts / sum(counts);
        p = p(p > 0);
        ent = -sum(p .* log2(p));
    else
        ent = 0;
    end
end

function analyzeCompressionDifferences(trainingData, testData, algorithm)
    fprintf('\n正在分析压缩差异...\n');

    all_diffs = [];
    relative_stds = [];

    for test_idx = 1:length(testData)
        test_author = testData(test_idx).author;
        if isempty(testData(test_idx).slices)
            continue;
        end

        test_slice = testData(test_idx).slices{1};
        fprintf('\n  测试作者: %s (切片大小: %.2f KB)\n', test_author, length(test_slice)/1024);

        diffs = [];
        for train_idx = 1:length(trainingData)
            train_text = trainingData(train_idx).text;
            combined = [train_text, test_slice];
            combined_size = compressData(combined, algorithm);
            train_size = trainingData(train_idx).trainCompressedSize;
            diff = combined_size - train_size;
            diffs = [diffs, diff]; %#ok<AGROW>
            all_diffs = [all_diffs, diff]; %#ok<AGROW>
        end

        fprintf('    压缩差异统计:\n');
        fprintf('      最小: %d 字节\n', min(diffs));
        fprintf('      最大: %d 字节\n', max(diffs));
        fprintf('      平均: %.0f 字节\n', mean(diffs));
        fprintf('      标准差: %.0f 字节\n', std(diffs));

        if mean(diffs) ~= 0
            relative_std = std(diffs) / abs(mean(diffs));
            relative_stds = [relative_stds, relative_std]; %#ok<AGROW>
            fprintf('      相对标准差: %.2f%%\n', relative_std*100);
            if relative_std < 0.1
                fprintf('      ⚠️  压缩差异太小（相对标准差 < 10%%）\n');
            end
        end
    end

    if ~isempty(all_diffs)
        fprintf('\n  全局压缩差异统计:\n');
        fprintf('    所有差异值数量: %d\n', length(all_diffs));
        fprintf('    最小值: %d 字节\n', min(all_diffs));
        fprintf('    最大值: %d 字节\n', max(all_diffs));
        fprintf('    平均值: %.0f 字节\n', mean(all_diffs));
        if mean(all_diffs) ~= 0
            fprintf('    全局相对标准差: %.2f%%\n', std(all_diffs)/abs(mean(all_diffs))*100);
        end

        if ~isempty(relative_stds)
            avg_relative_std = mean(relative_stds);
            fprintf('    平均相对标准差: %.2f%%\n', avg_relative_std*100);
            if avg_relative_std < 0.08
                fprintf('\n    ❌ 压缩差异极小 - 各模型文本特性接近\n');
            elseif avg_relative_std < 0.15
                fprintf('\n    ⚠️  压缩差异较小 - 分类难度高\n');
            end
        end
    end
end

%% ========== 训练数据准备 ==========
function trainingData = prepareTrainingData(folder, config)
    files = dir(fullfile(folder, '*.txt'));
    if length(files) < 2
        error('训练集需要至少2个作者的文本');
    end

    validateFileSizes(folder, files, config.requiredBytes);

    savePath = createSavePath(folder, config);
    config.savePath = savePath;

    trainingData = struct('author', {}, 'text', {}, 'sourceFile', {}, ...
                          'startPos', {}, 'fullTextLength', {}, 'trainCompressedSize', {});

    for i = 1:length(files)
        filePath = fullfile(folder, files(i).name);
        [author, fullText] = loadCompleteText(filePath);

        startPos = 1;
        endPos = min(startPos + config.trainSizeKB * 1024 - 1, length(fullText));
        trainText = fullText(startPos:endPos);

        saveTrainingSlice(author, trainText, savePath);

        trainingData(i).author = author;
        trainingData(i).text = trainText;
        trainingData(i).sourceFile = files(i).name;
        trainingData(i).startPos = startPos;
        trainingData(i).fullTextLength = length(fullText);

        fprintf('  ✓ %s: 提取训练数据 %dKB (位置: %d, 文件大小: %dKB)\n', ...
                author, config.trainSizeKB, startPos, floor(length(fullText)/1024));
    end

    fprintf('\n  [方案A] 预计算训练文本压缩大小...\n');
    tPre = tic;
    for i = 1:length(trainingData)
        trainingData(i).trainCompressedSize = compressData(trainingData(i).text, config.algorithm);
        fprintf('    ✓ %s: 压缩后 %d 字节\n', ...
                trainingData(i).author, trainingData(i).trainCompressedSize);
    end
    fprintf('  [方案A] 预计算完成，耗时 %.2f 秒\n', toc(tPre));
end

function [author, fullText] = loadCompleteText(filePath)
    [~, filename, ~] = fileparts(filePath);
    dashPos = strfind(filename, '-');
    if isempty(dashPos)
        error('文件名必须包含"-"分隔符，格式：作品名-作者.txt，当前文件：%s', filename);
    end
    lastDash = dashPos(end);
    author = strtrim(filename(lastDash+1:end));

    fid = fopen(filePath, 'r', 'n', 'UTF-8');
    if fid == -1
        error('无法打开文件: %s', filePath);
    end
    fullText = fread(fid, Inf, 'uint8=>char')';
    fclose(fid);
end

function saveTrainingSlice(author, text, savePath)
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    txtFile = fullfile(savePath, [author '_train.txt']);
    fid = fopen(txtFile, 'w', 'n', 'UTF-8');
    fwrite(fid, text, 'char');
    fclose(fid);
end

function savePath = createSavePath(folder, config)
    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST,TNOW1>
    saveRoot = fullfile(pwd, 'saved_train_sets');
    if ~exist(saveRoot, 'dir')
        mkdir(saveRoot);
    end
    [~, folderName] = fileparts(folder);
    savePath = fullfile(saveRoot, sprintf('%s_train%dKB_SEQ_%s', ...
                        folderName, config.trainSizeKB, timestamp));
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
end

function validateFileSizes(folder, files, requiredBytes)
    fprintf('\n=== 文件大小验证（每个文件需 ≥ %d KB）===\n', requiredBytes/1024);
    for i = 1:length(files)
        filePath = fullfile(folder, files(i).name);
        fileInfo = dir(filePath);
        fileSizeKB = ceil(fileInfo.bytes / 1024);
        requiredKB = requiredBytes / 1024;
        fprintf('  %s: %d KB ', files(i).name, fileSizeKB);
        if fileInfo.bytes < requiredBytes
            fprintf('❌\n');
            error('文件 "%s" 大小不足！需要 ≥ %d KB，实际 %d KB', ...
                  files(i).name, requiredKB, fileSizeKB);
        else
            fprintf('✅\n');
        end
    end
end

%% ========== 测试数据准备 ==========
function testData = prepareTestData(folder, config, trainingData)
    files = dir(fullfile(folder, '*.txt'));
    if length(files) < 1
        error('测试集文件夹为空');
    end

    testData = struct('author', {}, 'slices', {}, 'sourceFile', {});

    for i = 1:length(files)
        filePath = fullfile(folder, files(i).name);
        [author, fullText] = loadCompleteText(filePath);

        trainInfo = [];
        for j = 1:length(trainingData)
            if strcmp(strtrim(trainingData(j).author), strtrim(author))
                trainInfo = trainingData(j);
                break;
            end
        end

        testSlices = extractTestSlices(fullText, config, trainInfo, filePath);

        testData(i).author = author;
        testData(i).slices = testSlices;
        testData(i).sourceFile = files(i).name;

        fprintf('  ✓ %s: 提取 %d 个测试切片 (%dKB each)\n', author, length(testSlices), config.testSliceSizeKB);
    end
end

function testSlices = extractTestSlices(fullText, config, trainInfo, testFilePath)
    sliceBytes = config.testSliceSizeKB * 1024;
    requiredTestBytes = config.testSliceCount * sliceBytes;

    [~, testFileName, ~] = fileparts(testFilePath);
    dashPos = strfind(testFileName, '-');
    if ~isempty(dashPos)
        testAuthor = strtrim(testFileName(dashPos(end)+1:end));
    else
        testAuthor = strtrim(testFileName);
    end

    isSameAuthor = ~isempty(trainInfo) && strcmp(strtrim(trainInfo.author), testAuthor);

    if isSameAuthor
        startPos = trainInfo.startPos + config.trainSizeKB * 1024;
        fprintf('    ✓ 同一作者 → 从偏移 %d 字节处开始提取\n', startPos);
    else
        startPos = 1;
        fprintf('    ✗ 不同作者（顺序模式） → 从偏移 1 字节处开始\n');
    end

    startPos = max(1, startPos);

    if startPos + requiredTestBytes - 1 > length(fullText)
        error('文件 "%s" 长度不足，无法提取测试切片。需要 %d 字节，从 %d 开始可用 %d 字节', ...
              testFilePath, requiredTestBytes, startPos, length(fullText) - startPos + 1);
    end

    testBlock = fullText(startPos:startPos + requiredTestBytes - 1);

    testSlices = cell(1, config.testSliceCount);
    for k = 1:config.testSliceCount
        s = (k-1)*sliceBytes + 1;
        e = k*sliceBytes;
        testSlices{k} = testBlock(s:e);
    end
end

%% ========== 分类执行 ==========
function [resultTable, failStats] = performClassification(trainingData, testData, algorithm)
    authors = {trainingData.author};
    numAuthors = length(authors);
    resultTable = zeros(numAuthors);

    totalSlices = 0;
    failedSlices = 0;
    failedDetails = {};

    for i = 1:length(testData)
        testAuthorNorm = strtrim(testData(i).author);
        trueIdx = 0;
        for k = 1:numAuthors
            if strcmp(strtrim(authors{k}), testAuthorNorm)
                trueIdx = k;
                break;
            end
        end

        if trueIdx == 0
            warning('测试作者 %s 不在训练集中，整个作者跳过', testData(i).author);
            continue;
        end

        slices = testData(i).slices;

        for j = 1:length(slices)
            totalSlices = totalSlices + 1;
            try
                predAuthor = classifySingleSlice(trainingData, slices{j}, algorithm);
                predAuthorNorm = strtrim(predAuthor);

                predIdx = 0;
                for k = 1:numAuthors
                    if strcmp(strtrim(authors{k}), predAuthorNorm)
                        predIdx = k;
                        break;
                    end
                end

                if predIdx == 0
                    failedSlices = failedSlices + 1;
                    failedDetails{end+1} = sprintf('%s 切片%d: 预测作者"%s"不在训练集', ...
                                                   testData(i).author, j, predAuthor); %#ok<AGROW>
                    continue;
                end

                resultTable(trueIdx, predIdx) = resultTable(trueIdx, predIdx) + 1;

            catch ME
                failedSlices = failedSlices + 1;
                failedDetails{end+1} = sprintf('%s 切片%d: %s', ...
                                               testData(i).author, j, ME.message); %#ok<AGROW>
                continue;
            end
        end
    end

    failStats = struct('totalSlices', totalSlices, ...
                       'failedSlices', failedSlices, ...
                       'failedRate', (totalSlices > 0) * failedSlices/max(totalSlices,1), ...
                       'failedDetails', {failedDetails});
end

function bestAuthor = classifySingleSlice(trainingData, testSlice, algorithm)
    bestAuthor = '';
    minDiff = Inf;

    for i = 1:length(trainingData)
        diff = calculateCompressionDiff(trainingData(i).text, testSlice, algorithm, ...
                                        trainingData(i).trainCompressedSize);
        if diff < minDiff
            minDiff = diff;
            bestAuthor = trainingData(i).author;
        end
    end

    if isempty(bestAuthor)
        error('无法找到最佳匹配作者');
    end
end

function diff = calculateCompressionDiff(trainText, testSlice, algorithm, precomputedTrainSize)
    if nargin < 4 || isempty(precomputedTrainSize)
        precomputedTrainSize = compressData(trainText, algorithm);
    end
    combined = [trainText, testSlice];
    combinedSize = compressData(combined, algorithm);
    diff = combinedSize - precomputedTrainSize;
end

%% ========== 压缩算法 ==========
function verifyCompressionTools(algorithm)
    if algorithm ~= 2 && algorithm ~= 6
        error('精简版只支持 LZMA(2) 和 PPMd(6)');
    end
    if ispc
        [status, ~] = system('where 7z > nul 2>&1');
    else
        [status, ~] = system('which 7z > /dev/null 2>&1');
    end
    if status ~= 0
        error('7-Zip未安装! 请安装7-Zip并添加到PATH');
    end
    if algorithm == 2
        fprintf('✓ LZMA压缩（通过7-Zip）可用\n');
    else
        fprintf('✓ PPMd压缩（通过7-Zip）可用\n');
    end
end

function compressedSize = compressData(data, algorithm)
    if algorithm == 2
        compressedSize = compressWith7Zip(data, 'lzma');
    elseif algorithm == 6
        compressedSize = compressWith7Zip(data, 'ppmd');
    else
        error('不支持的压缩算法: %d', algorithm);
    end
end

function compressedSize = compressWith7Zip(data, method)
    tempFile = [tempname '.tmp'];
    if strcmp(method, 'lzma')
        compressedFile = [tempFile '.7z'];
    else
        compressedFile = [tempFile '.ppm7z'];
    end

    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, compressedFile)); %#ok<NASGU>

    fid = fopen(tempFile, 'wb');
    if fid == -1
        error('无法创建临时文件');
    end
    fwrite(fid, data, 'uint8');
    fclose(fid);

    if ispc
        nullRedir = ' > nul 2>&1';
    else
        nullRedir = ' > /dev/null 2>&1';
    end

    if strcmp(method, 'lzma')
        cmd = sprintf('7z a -mmt=off -mx=1 -y "%s" "%s" %s', compressedFile, tempFile, nullRedir);
    else
        cmd = sprintf('7z a -mmt=off -mx=1 -m0=PPMd -y "%s" "%s" %s', compressedFile, tempFile, nullRedir);
    end

    status = system(cmd);
    if status ~= 0
        error('7-Zip压缩失败');
    end

    maxWait = 10;
    waitInt = 0.1;
    elapsedTime = 0;
    while ~exist(compressedFile, 'file') && elapsedTime < maxWait
        pause(waitInt);
        elapsedTime = elapsedTime + waitInt;
    end

    if ~exist(compressedFile, 'file')
        error('7-Zip压缩失败');
    end

    dirInfo = dir(compressedFile);
    compressedSize = dirInfo.bytes;
end

function cleanupTempFiles(tempFile, compressedFile)
    if exist(tempFile, 'file')
        try delete(tempFile); catch; end %#ok<CTCH>
    end
    if exist(compressedFile, 'file')
        try delete(compressedFile); catch; end %#ok<CTCH>
    end
end

function cleanupTempFolder()
    tempDir = tempdir;
    tempFiles = dir(fullfile(tempDir, 'tp*.tmp*'));
    for i = 1:length(tempFiles)
        f = fullfile(tempDir, tempFiles(i).name);
        if exist(f, 'file')
            try delete(f); catch; end %#ok<CTCH>
        end
    end
end

%% ========== 结果输出 ==========
function printClassificationReport(table, authors)
    fprintf('\n=== 分类结果矩阵 ===\n');
    fprintf('%15s', ' ');
    for i = 1:length(authors)
        fprintf('%12s', authors{i});
    end
    fprintf('\n');

    for i = 1:length(authors)
        fprintf('%12s: ', authors{i});
        for j = 1:length(authors)
            fprintf('%8d', table(i,j));
        end
        fprintf('\n');
    end

    total = sum(table(:));
    correct = sum(diag(table));

    if total == 0
        fprintf('\n⚠️  混淆矩阵为空，无有效分类结果\n');
        return;
    end

    fprintf('\n=== 统计（仅基于有效切片）===\n');
    fprintf('有效切片总数: %d\n', total);
    fprintf('正确分类数: %d\n', correct);
    fprintf('准确率: %.2f%% (%d/%d)\n', correct/total*100, correct, total);

    rowTotals = sum(table, 2);
    colTotals = sum(table, 1);
    zeroRows = find(rowTotals == 0);
    zeroCols = find(colTotals == 0);
    if ~isempty(zeroRows)
        fprintf('⚠️  以下作者无任何有效预测（行全零）: %s\n', ...
                strjoin(authors(zeroRows), ', '));
    end
    if ~isempty(zeroCols)
        fprintf('⚠️  以下作者从未被预测（列全零）: %s\n', ...
                strjoin(authors(zeroCols), ', '));
    end

    cramerV = calculateCramersV(table);
    fprintf('Cramer''s V系数（剔除退化类别后）: %.4f\n', cramerV);
    interpretCramersV(cramerV);
end

%% ========== 方案2：剔除零行零列后计算 Cramér's V ==========
function cramerV = calculateCramersV(contingencyTable)
    rowTotals = sum(contingencyTable, 2);
    colTotals = sum(contingencyTable, 1);
    tbl = contingencyTable(rowTotals > 0, colTotals > 0);

    if isempty(tbl) || size(tbl, 1) < 2 || size(tbl, 2) < 2
        cramerV = 0;
        return;
    end

    [chi2, ~, ~] = computeChiSquareFromTable(tbl);
    n = sum(tbl(:));
    k = min(size(tbl));

    if n == 0 || k == 1
        cramerV = 0;
    else
        cramerV = sqrt(chi2 / (n * (k - 1)));
    end
end

%% ========== 卡方检验 ==========
function [chi2, p, df] = computeChiSquareFromTable(observed)
    rowTotals = sum(observed, 2);
    colTotals = sum(observed, 1);
    n = sum(observed(:));

    if n == 0
        chi2 = 0; p = 1; df = 0; return;
    end

    validRows = rowTotals > 0;
    validCols = colTotals > 0;
    observed = observed(validRows, validCols);

    if isempty(observed) || size(observed,1) < 2 || size(observed,2) < 2
        chi2 = 0; p = 1; df = 0; return;
    end

    rowTotals = sum(observed, 2);
    colTotals = sum(observed, 1);
    n = sum(observed(:));

    expected = (rowTotals * colTotals) / n;
    expected(expected < 1e-10) = 1e-10;

    chi2 = sum((observed - expected).^2 ./ expected, 'all');
    [r, c] = size(observed);
    df = (r - 1) * (c - 1);

    if df > 0
        p = 1 - chi2cdf(chi2, df);
    else
        p = 1;
    end
end

function interpretCramersV(v)
    fprintf('\nCramer''s V系数解释:\n');
    if v < 0.1
        fprintf('  %.4f - 无关联或极弱关联（风格一致性最好）\n', v);
    elseif v < 0.3
        fprintf('  %.4f - 弱关联（风格一致性较好）\n', v);
    elseif v < 0.5
        fprintf('  %.4f - 中等关联（风格差异中等）\n', v);
    else
        fprintf('  %.4f - 强关联（风格差异较大）\n', v);
    end
end

%% ========== 保存结果 ==========
function saveResults(resultTable, trainFolder, config, authors, failStats, subsetResults)
    [~, folderName] = fileparts(trainFolder);
    if config.algorithm == 2
        algoName = 'LZMA';
    else
        algoName = 'PPMd';
    end

    excelName = sprintf('Results_%s_%s_T%dKB_S%dKB_N%d.xlsx', ...
                      folderName, algoName, ...
                      config.trainSizeKB, config.testSliceSizeKB, config.testSliceCount);

    resultCell = cell(size(resultTable, 1) + 1, size(resultTable, 2) + 1);
    resultCell{1,1} = 'True\Pred';
    for i = 1:length(authors)
        resultCell{1, i+1} = authors{i};
        resultCell{i+1, 1} = authors{i};
        for j = 1:length(authors)
            resultCell{i+1, j+1} = resultTable(i,j);
        end
    end

    try
        writecell(resultCell, excelName);
        fprintf('Excel文件已保存: %s\n', excelName);
    catch
        xlswrite(excelName, resultCell);
        fprintf('Excel文件已保存(使用xlswrite): %s\n', excelName);
    end

    total = sum(resultTable(:));
    if total > 0
        accuracy = sum(diag(resultTable)) / total * 100;
    else
        accuracy = 0;
    end

    cramerV_raw = calculateCramersVRaw(resultTable);
    cramerV_adj = calculateCramersV(resultTable);
    [~, pVal, ~] = computeChiSquareFromTable(resultTable);

    rowTotals = sum(resultTable, 2);
    colTotals = sum(resultTable, 1);
    zeroRows = find(rowTotals == 0);
    zeroCols = find(colTotals == 0);

    statsFile = strrep(excelName, '.xlsx', '_stats.txt');
    fid = fopen(statsFile, 'w');
    fprintf(fid, '=== 统计报告 ===\n');
    fprintf(fid, '生成时间: %s\n', datestr(now)); %#ok<DATST,TNOW1>
    fprintf(fid, '压缩算法: %s\n', algoName);
    fprintf(fid, '训练集大小: %d KB\n', config.trainSizeKB);
    fprintf(fid, '测试切片大小: %d KB\n', config.testSliceSizeKB);
    fprintf(fid, '每个作者切片数: %d\n', config.testSliceCount);
    fprintf(fid, '截取方式: 顺序\n');
    fprintf(fid, 'minSlices 阈值: %d\n', config.minSlices);
    if isfinite(config.fixedSlicesPerAuthor)
        fprintf(fid, '固定每作者切片数: %d\n', config.fixedSlicesPerAuthor);
    end
    fprintf(fid, '\n=== 切片统计 ===\n');
    fprintf(fid, '总切片数: %d\n', failStats.totalSlices);
    fprintf(fid, '失败切片数: %d\n', failStats.failedSlices);
    fprintf(fid, '失败率: %.2f%%\n', failStats.failedRate*100);
    fprintf(fid, '有效切片数: %d\n', total);
    fprintf(fid, '\n=== 分类统计 ===\n');
    fprintf(fid, '准确率: %.2f%%\n', accuracy);
    fprintf(fid, 'Cramer''s V (原始，含退化类别): %.4f\n', cramerV_raw);
    fprintf(fid, 'Cramer''s V (修正，剔除零行列): %.4f\n', cramerV_adj);
    fprintf(fid, '卡方检验p值: %.4e\n', pVal);
    fprintf(fid, '\n=== 退化类别 ===\n');
    if ~isempty(zeroRows)
        fprintf(fid, '行全零（无有效预测）: %s\n', strjoin(authors(zeroRows), ', '));
    else
        fprintf(fid, '行全零: 无\n');
    end
    if ~isempty(zeroCols)
        fprintf(fid, '列全零（从未被预测）: %s\n', strjoin(authors(zeroCols), ', '));
    else
        fprintf(fid, '列全零: 无\n');
    end

    fprintf(fid, '\n=== 混淆矩阵 ===\n');
    fprintf(fid, '行=真实作者，列=预测作者\n\n');
    fprintf(fid, 'True\\Pred\t');
    for i = 1:length(authors)
        fprintf(fid, '%s\t', authors{i});
    end
    fprintf(fid, '\n');
    for i = 1:length(authors)
        fprintf(fid, '%s\t', authors{i});
        for j = 1:length(authors)
            fprintf(fid, '%d\t', resultTable(i,j));
        end
        fprintf(fid, '\n');
    end

    % ---- 追加：全局枚举组合结果 ----
    fprintf(fid, '\n\n=== 全局枚举组合结果（已剔除 n < %d）===\n', config.minSlices);
    writeEnumeration(fid, subsetResults.rank2, 'worst2 候选 (V降序)');
    writeEnumeration(fid, subsetResults.rank4, 'top4 候选 (V升序)');
    writeEnumeration(fid, subsetResults.rank3, 'top3 候选 (V升序)');

    fclose(fid);

    fprintf('统计报告已保存: %s\n', statsFile);
    fprintf('\n结果已保存至当前目录\n');
end

function writeEnumeration(fid, recList, titleStr)
    fprintf(fid, '\n【%s】\n', titleStr);
    fprintf(fid, '%-40s\t%8s\t%6s\t%4s\n', '组合', 'V', 'n', 'k');
    if isempty(recList)
        fprintf(fid, '(无有效组合)\n');
        return;
    end
    for i = 1:numel(recList)
        r = recList(i);
        fprintf(fid, '%-40s\t%8.4f\t%6d\t%4d\n', ...
                strjoin(r.authors, '+'), r.V, r.n, r.k);
    end
end

%% ========== 原始 V（含退化类别，用于对比报告） ==========
function cramerV = calculateCramersVRaw(contingencyTable)
    [chi2, ~, ~] = computeChiSquareRaw(contingencyTable);
    n = sum(contingencyTable(:));
    k = min(size(contingencyTable));
    if n == 0 || k == 1
        cramerV = 0;
    else
        cramerV = sqrt(chi2 / (n * (k - 1)));
    end
end

function [chi2, p, df] = computeChiSquareRaw(observed)
    rowTotals = sum(observed, 2);
    colTotals = sum(observed, 1);
    n = sum(observed(:));

    if n == 0
        chi2 = 0; p = 1; df = 0; return;
    end

    expected = (rowTotals * colTotals) / n;
    expected(expected < 1e-10) = 1e-10;

    chi2 = sum((observed - expected).^2 ./ expected, 'all');
    [r, c] = size(observed);
    df = (r - 1) * (c - 1);

    if df > 0
        p = 1 - chi2cdf(chi2, df);
    else
        p = 1;
    end
end