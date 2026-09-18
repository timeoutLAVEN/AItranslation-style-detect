function main()
    % 主函数 - 基于压缩的作者识别系统（单文件夹模式）
    % 所有作者文件放在同一个文件夹中，从每个文件中连续切分训练和测试（不重叠）
    rng(2023, 'twister');
    try
        cleanupTempFolder();
        
        % === 修改：只用一个文件夹 ===
        dataFolder = 'C:\Users\han\Desktop\distiguish';  % 请替换为您的实际文件夹路径
        
        config = getConfiguration();
        
        % 阶段1+2：同时准备训练和测试数据（单文件夹切分）
        fprintf('\n=== 阶段1+2：从同一文件切分训练和测试 ===\n');
        [trainingData, testData] = prepareBothData(dataFolder, config);
        
        % 阶段3a：找出翻译效果最差的 2 个模型（worst2 = V最大）
        fprintf('\n=== 阶段3a：寻找翻译效果最差的 2 个模型（worst2 = V最大）===\n');
        [worst2_authors, worst2_V, ~] = findWorst2Models(trainingData, testData, config.algorithm);
        
        % 阶段3b：计算删掉 worst2 后的 top4 V
        fprintf('\n=== 阶段3b：计算删掉 worst2 后的 top4 ===\n');
        authors_all = {trainingData.author};
        worst2_indices = find(ismember(authors_all, worst2_authors));
        keep_idx_top4 = setdiff(1:length(authors_all), worst2_indices);
        trainingData_top4 = trainingData(keep_idx_top4);
        testData_top4 = testData(ismember({testData.author}, {authors_all{keep_idx_top4}}));
        resultTable_top4 = performClassification(trainingData_top4, testData_top4, config.algorithm);
        top4_V = calculateCramersV(resultTable_top4);
        fprintf('top4 V 值: %.4f\n', top4_V);
        
        % 阶段3c：计算 top3 V（从 top4 里再删掉使 V 最小的 1 个）
        fprintf('\n=== 阶段3c：计算 top3 V（从 top4 里删掉使 V 最小的 1 个）===\n');
        top4_authors = {authors_all{keep_idx_top4}};
        minV_top3 = Inf;
        best_remove_idx = 0;
        best_remove_author = '';
        fprintf('尝试从 top4 里删掉一个（找哪个删掉后 V 最小）:\n');
        fprintf('%-15s | V值\n', '删掉的人员');
        fprintf(repmat('-', 1, 30));
        fprintf('\n');
        for i = 1:length(keep_idx_top4)
            keep_idx_top3 = setdiff(1:length(keep_idx_top4), i);
            trainingData_top3_temp = trainingData_top4(keep_idx_top3);
            testData_top3_temp = testData_top4(ismember({testData_top4.author}, {top4_authors{keep_idx_top3}}));
            resultTable_top3_temp = performClassification(trainingData_top3_temp, testData_top3_temp, config.algorithm);
            v_temp = calculateCramersV(resultTable_top3_temp);
            fprintf('%-15s | %.4f', top4_authors{i}, v_temp);
            if v_temp < minV_top3
                minV_top3 = v_temp;
                best_remove_idx = i;
                best_remove_author = top4_authors{i};
                fprintf(' ← 目前最小（最优）\n');
            else
                fprintf('\n');
            end
        end
        keep_idx_top3_final = setdiff(1:length(keep_idx_top4), best_remove_idx);
        trainingData_top3 = trainingData_top4(keep_idx_top3_final);
        testData_top3 = testData_top4(ismember({testData_top4.author}, {top4_authors{keep_idx_top3_final}}));
        resultTable_top3 = performClassification(trainingData_top3, testData_top3, config.algorithm);
        top3_V = calculateCramersV(resultTable_top3);
        fprintf('✓ top3 V 值: %.4f (删掉了 %s)\n', top3_V, best_remove_author);
        fprintf('  保留的 3 个模型: %s\n', strjoin({top4_authors{keep_idx_top3_final}}, ' + '));
        
        % 阶段4：执行完整的 6 人分类
        fprintf('\n=== 阶段4：执行完整的 6 人分类 ===\n');
        resultTable = performClassification(trainingData, testData, config.algorithm);
        
        % 阶段4.5：诊断分析
        fprintf('\n=== 阶段4.5：诊断分析 ===\n');
        diagnoseAbnormalCramersV(trainingData, testData, config.algorithm, 'CTR (中→俄)');
        
        % 阶段5：保存结果
        fprintf('\n=== 阶段5：保存结果 ===\n');
        authors = {trainingData.author};
        saveResults(resultTable, dataFolder, config, authors);  % 保存使用 dataFolder
        
        % 最终汇总
        fprintf('\n');
        fprintf('╔════════════════════════════════════════════╗\n');
        fprintf('║            分析结果汇总                    ║\n');
        fprintf('║  V越小=翻译效果好, V越大=翻译效果差       ║\n');
        fprintf('╚════════════════════════════════════════════╝\n');
        six_V = calculateCramersV(resultTable);
        fprintf('6人全部         V = %.4f (所有模型混合)\n', six_V);
        fprintf('worst2 (%s + %s): V = %.4f (翻译效果最差的2个)\n', worst2_authors{1}, worst2_authors{2}, worst2_V);
        fprintf('top4  (删掉worst2)  : V = %.4f (删掉效果最差的2个)\n', top4_V);
        fprintf('top3  (再删掉%s)    : V = %.4f (最优的3个模型)\n', best_remove_author, top3_V);
        fprintf('\n解释：V值越小越好，说明翻译风格越稳定一致\n');
        fprintf('=== 分析完成! ===\n');
        
    catch ME
        fprintf('\n=== 程序执行失败 ===\n');
        fprintf('错误信息: %s\n', ME.message);
        fprintf('错误位置: %s\n', ME.stack(1).name);
        cleanupTempFolder();
        rethrow(ME);
    end
end

%% ========== 统一数据准备（单文件夹模式） ==========
function [trainingData, testData] = prepareBothData(folder, config)
    % 从同一个文件夹中读取所有文本，为每个作者连续切分训练和测试（不重叠）
    files = dir(fullfile(folder, '*.txt'));
    if length(files) < 2
        error('需要至少2个作者的文本');
    end
    
    % 验证文件大小（训练+测试总大小）
    requiredBytes = (config.trainSizeKB + config.testSliceCount * config.testSliceSizeKB) * 1024;
    validateFileSizes(folder, files, requiredBytes);
    
    % 创建保存目录（仅保存训练部分）
    savePath = createSavePath(folder, config);
    
    % 设置随机种子
    if config.isRandom
        rng(2023, 'twister');
    end
    
    trainBytes = config.trainSizeKB * 1024;
    sliceBytes = config.testSliceSizeKB * 1024;
    totalTestBytes = config.testSliceCount * sliceBytes;
    
    trainingData = struct('author', {}, 'text', {}, 'sourceFile', {}, 'startPos', {}, 'fullTextLength', {});
    testData = struct('author', {}, 'slices', {}, 'sourceFile', {});
    
    for i = 1:length(files)
        filePath = fullfile(folder, files(i).name);
        [author, fullText] = loadCompleteText(filePath);
        
        % 确定训练起始位置
        if config.isRandom
            maxTrainStart = max(1, length(fullText) - trainBytes - totalTestBytes + 1);
            if maxTrainStart < 1
                error('文件 "%s" 长度不足，无法随机选取训练和测试', files(i).name);
            end
            trainStart = randi([1, maxTrainStart]);
        else
            trainStart = 1;
        end
        
        % 提取训练部分
        trainText = fullText(trainStart:trainStart + trainBytes - 1);
        
        % 提取测试部分（紧接训练之后，不重叠）
        testStart = trainStart + trainBytes;
        if testStart + totalTestBytes - 1 > length(fullText)
            error('文件 "%s" 剩余部分不足，无法提取 %d 个 %dKB 测试切片', ...
                   files(i).name, config.testSliceCount, config.testSliceSizeKB);
        end
        testBlock = fullText(testStart:testStart + totalTestBytes - 1);
        
        % 分割测试块为切片
        testSlices = cell(1, config.testSliceCount);
        for k = 1:config.testSliceCount
            s = (k-1)*sliceBytes + 1;
            e = k*sliceBytes;
            testSlices{k} = testBlock(s:e);
        end
        
        % 保存训练数据（可选）
        saveTrainingSlice(author, trainText, savePath);
        
        % 存储到结构体
        trainingData(i).author = author;
        trainingData(i).text = trainText;
        trainingData(i).sourceFile = files(i).name;
        trainingData(i).startPos = trainStart;
        trainingData(i).fullTextLength = length(fullText);
        
        testData(i).author = author;
        testData(i).slices = testSlices;
        testData(i).sourceFile = files(i).name;
        
        fprintf('  ✓ %s: 训练 %dKB (偏移 %d) + %d 个测试切片 (偏移 %d)\n', ...
                author, config.trainSizeKB, trainStart, config.testSliceCount, testStart);
    end
end

%% ========== 配置管理 ==========
function config = getConfiguration()
    % 集中管理所有配置参数
    config.trainSizeKB = validateInput('请输入训练集大小(KB)：', 10, 1000);
    config.testSliceSizeKB = validateInput('请输入测试切片大小(KB)：', 1, config.trainSizeKB);
    
    % 核心公式：切片数量 = 训练集大小 / 切片大小
    config.testSliceCount = floor(config.trainSizeKB / config.testSliceSizeKB);
    if config.testSliceCount < 1
        error('错误：切片大小不能大于或等于训练集大小！');
    end
    
    config.isRandom = selectRandomOption();
    config.algorithm = selectCompressionAlgorithm();
    config.requiredBytes = (config.trainSizeKB + config.testSliceCount * config.testSliceSizeKB) * 1024;
    
    fprintf('自动计算：每个作者使用 %d 个 %dKB 测试切片（每个文件需 ≥ %dKB）\n', ...
            config.testSliceCount, config.testSliceSizeKB, config.requiredBytes/1024);
    
    % 验证压缩工具
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

function isRandom = selectRandomOption()
    fprintf('\n训练集选择方式:\n');
    fprintf('1) 顺序选取 - 从文本开头连续截取\n');
    fprintf('2) 随机选取 - 从随机位置截取\n');
    while true
        choice = input('请选择截取方式 (1/2): ');
        if ismember(choice, [1, 2])
            isRandom = (choice == 2);
            return;
        end
        fprintf('输入错误，请重新输入\n');
    end
end

function compressionAlgorithm = selectCompressionAlgorithm()
    fprintf('\n可用压缩算法:\n');
    fprintf('1) GZIP (MATLAB内置)\n');
    fprintf('2) LZMA (7-Zip)\n');
    fprintf('3) BZIP2 (需外部工具)\n');
    fprintf('4) LZ4 (需外部工具)\n');
    fprintf('5) Zstandard (需外部工具)\n');
    fprintf('6) PPMd (7-Zip PPMd实现)\n');
    while true
        compressionAlgorithm = input('请选择压缩算法 (1-6)：');
        if ismember(compressionAlgorithm, 1:6)
            return;
        end
        fprintf('输入错误，请输入1-6之间的数字\n');
    end
end

%% ========== 异常诊断函数 ==========
function diagnoseAbnormalCramersV(trainingData, testData, algorithm, languagePair)
    % 诊断异常小的 Cramér V 值
    
    fprintf('\n╔════════════════════════════════════════════╗\n');
    fprintf('║  诊断异常 Cramér V 值: %s\n', languagePair);
    fprintf('╚════════════════════════════════════════════╝\n');
    
    % 1. 执行分类
    resultTable = performClassification(trainingData, testData, algorithm);
    cramerV = calculateCramersV(resultTable);
    
    fprintf('\n【基础统计】\n');
    fprintf('  Cramér V 值: %.4f\n', cramerV);
    fprintf('  样本总数: %d\n', sum(resultTable(:)));
    fprintf('  准确率: %.2f%%\n', sum(diag(resultTable))/sum(resultTable(:))*100);
    
    % 2. 分析混淆矩阵的特性
    fprintf('\n【混淆矩阵分析】\n');
    
    % 计算卡方值
    [chi2, p, df] = computeChiSquareFromTable(resultTable);
    fprintf('  卡方值 (χ²): %.4f\n', chi2);
    fprintf('  自由度: %d\n', df);
    fprintf('  P 值: %.4e\n', p);
    
    if p > 0.05
        fprintf('  ⚠️  P 值 > 0.05：两个变量独立（无显著关联）\n');
        fprintf('     这意味着预测和真实标签没有显著相关性！\n');
    else
        fprintf('  ✓ P 值 < 0.05：两个变量有显著关联\n');
    end
    
    % 3. 分析行的分布
    fprintf('\n【行（真实标签）的分布】\n');
    rowTotals = sum(resultTable, 2);
    authors = {trainingData.author};
    for i = 1:length(authors)
        fprintf('  %s: %d (%.1f%%)\n', authors{i}, rowTotals(i), rowTotals(i)/sum(rowTotals)*100);
    end
    
    % 4. 分析列的分布
    fprintf('\n【列（预测标签）的分布】\n');
    colTotals = sum(resultTable, 1);
    for j = 1:length(authors)
        fprintf('  %s: %d (%.1f%%)\n', authors{j}, colTotals(j), colTotals(j)/sum(colTotals)*100);
    end
    
    % 5. 检查对角线（正确预测）
    fprintf('\n【对角线分析（正确预测）】\n');
    diag_values = diag(resultTable);
    for i = 1:length(authors)
        correct_rate = diag_values(i) / rowTotals(i) * 100;
        fprintf('  %s: %d/%d (%.1f%%)\n', authors{i}, diag_values(i), rowTotals(i), correct_rate);
    end
    
    % 6. 检查是否接近均匀分布
    fprintf('\n【均匀分布检验】\n');
    n_authors = length(authors);
    expected_per_cell = sum(resultTable(:)) / (n_authors * n_authors);
    fprintf('  如果完全随机，每个格子应该约有 %.0f 个样本\n', expected_per_cell);
    fprintf('  实际混淆矩阵:\n');
    
    % 打印混淆矩阵
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
    
    % 检查是否过于均匀
    variance_from_uniform = var(resultTable(:));
    fprintf('\n  矩阵中所有值的方差: %.2f\n', variance_from_uniform);
    
    if variance_from_uniform < 10
        fprintf('  ⚠️  矩阵值过于均匀！可能是完全随机分类\n');
        fprintf('     这会导致 Cramér V 非常小\n');
    else
        fprintf('  ✓ 矩阵分布合理\n');
    end
    
    % 7. 检查是否存在特定的预测偏向
    fprintf('\n【预测偏向分析】\n');
    for j = 1:n_authors
        most_predicted_for = 0;
        most_predicted_count = 0;
        for i = 1:n_authors
            if resultTable(i,j) > most_predicted_count
                most_predicted_count = resultTable(i,j);
                most_predicted_for = i;
            end
        end
        fprintf('  预测为 %s: 最多来自真实 %s (%d 个)\n', ...
                authors{j}, authors{most_predicted_for}, most_predicted_count);
    end
    
    % 8. 压缩差异分析
    fprintf('\n【压缩差异分析】\n');
    analyzeCompressionDifferences(trainingData, testData, algorithm);
    
    % 9. 对标分析和建议
    fprintf('\n【对标分析和诊断结论】\n');
    fprintf('  当前 V 值: %.4f\n', cramerV);
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

function analyzeCompressionDifferences(trainingData, testData, algorithm)
    % 分析为什么压缩差异这么小
    
    fprintf('\n正在分析压缩差异...\n');
    
    % 统计所有测试切片的压缩差异
    all_diffs = [];
    relative_stds = [];
    
    for test_idx = 1:length(testData)
        test_author = testData(test_idx).author;
        if isempty(testData(test_idx).slices)
            continue;
        end
        
        % 取第一个测试切片进行分析
        test_slice = testData(test_idx).slices{1};
        
        fprintf('\n  测试作者: %s (切片大小: %.2f KB)\n', test_author, length(test_slice)/1024);
        
        % 计算与每个训练作者的压缩差异
        diffs = [];
        authors = {trainingData.author};
        
        for train_idx = 1:length(trainingData)
            train_author = trainingData(train_idx).author;
            train_text = trainingData(train_idx).text;
            
            combined = [train_text, test_slice];
            combined_size = compressData(combined, algorithm);
            train_size = compressData(train_text, algorithm);
            diff = combined_size - train_size;
            
            diffs = [diffs, diff];
            all_diffs = [all_diffs, diff];
        end
        
        % 分析差异的离散度
        fprintf('    压缩差异统计:\n');
        fprintf('      最小: %d 字节\n', min(diffs));
        fprintf('      最大: %d 字节\n', max(diffs));
        fprintf('      平均: %.0f 字节\n', mean(diffs));
        fprintf('      标准差: %.0f 字节\n', std(diffs));
        
        relative_std = std(diffs) / mean(diffs);
        relative_stds = [relative_stds, relative_std];
        fprintf('      相对标准差: %.2f%%\n', relative_std*100);
        
        if relative_std < 0.1
            fprintf('      ⚠️  压缩差异太小（相对标准差 < 10%%）\n');
            fprintf('         各作者文本的压缩特性接近，难以区分！\n');
        end
    end
    
    % 全局统计
    if ~isempty(all_diffs)
        fprintf('\n  全局压缩差异统计:\n');
        fprintf('    所有差异值数量: %d\n', length(all_diffs));
        fprintf('    最小值: %d 字节\n', min(all_diffs));
        fprintf('    最大值: %d 字节\n', max(all_diffs));
        fprintf('    平均值: %.0f 字节\n', mean(all_diffs));
        fprintf('    全局相对标准差: %.2f%%\n', std(all_diffs)/mean(all_diffs)*100);
        
        avg_relative_std = mean(relative_stds);
        fprintf('    平均相对标准差: %.2f%%\n', avg_relative_std*100);
        
        if avg_relative_std < 0.08
            fprintf('\n    ❌ 压缩差异极小 - 各模型文本特性接近\n');
            fprintf('       这是导致 V 值很小的主要原因\n');
        elseif avg_relative_std < 0.15
            fprintf('\n    ⚠️  压缩差异较小 - 分类难度高\n');
        end
    end
end

%% ========== 训练数据准备（备用，不再被主函数调用） ==========
function trainingData = prepareTrainingData(folder, config)
    % 只负责准备训练数据
    files = dir(fullfile(folder, '*.txt'));
    if length(files) < 2
        error('训练集需要至少2个作者的文本');
    end
    
    % 验证文件大小
    validateFileSizes(folder, files, config.requiredBytes);
    
    % 创建保存目录
    savePath = createSavePath(folder, config);
    config.savePath = savePath;  % 保存到config供后续使用
    
    % 设置随机种子
    if config.isRandom
        rng(2023, 'twister');
    end
    
    % 为每个作者准备训练数据
    trainingData = struct('author', {}, 'text', {}, 'sourceFile', {}, 'startPos', {}, 'fullTextLength', {});
    
    for i = 1:length(files)
        filePath = fullfile(folder, files(i).name);
        
        % 读取完整文本
        [author, fullText] = loadCompleteText(filePath);
        
        % 提取训练片段
        [trainText, startPos] = extractTrainingSlice(fullText, config.trainSizeKB * 1024, config.isRandom);
        
        % 保存训练数据
        saveTrainingSlice(author, trainText, savePath);
        
        % 存储到结构体
        trainingData(i).author = author;
        trainingData(i).text = trainText;
        trainingData(i).sourceFile = files(i).name;
        trainingData(i).startPos = startPos;
        trainingData(i).fullTextLength = length(fullText);
        
        fprintf('  ✓ %s: 提取训练数据 %dKB (位置: %d, 文件大小: %dKB)\n', ...
                author, config.trainSizeKB, startPos, floor(length(fullText)/1024));
    end
end

function [author, fullText] = loadCompleteText(filePath)
    % 只读取完整文本，不进行截取
    [~, filename, ~] = fileparts(filePath);
    dashPos = strfind(filename, '-');
    if isempty(dashPos)
        error('文件名必须包含"-"分隔符，格式：作品名-作者.txt，当前文件：%s', filename);
    end
    lastDash = dashPos(end);
    author = filename(lastDash+1:end);
    
    fid = fopen(filePath, 'r', 'n', 'UTF-8');
    if fid == -1
        error('无法打开文件: %s', filePath);
    end
    fullText = fread(fid, Inf, 'uint8=>char')';
    fclose(fid);
end

function [trainText, startPos] = extractTrainingSlice(fullText, trainBytes, isRandom)
    % 从完整文本中提取训练片段
    if isRandom
        maxStart = max(1, length(fullText) - trainBytes + 1);
        startPos = randi([1, maxStart]);
    else
        startPos = 1;
    end
    
    endPos = min(startPos + trainBytes - 1, length(fullText));
    trainText = fullText(startPos:endPos);
end

function saveTrainingSlice(author, text, savePath)
    % 保存训练数据到文件
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    txtFile = fullfile(savePath, [author '_train.txt']);
    fid = fopen(txtFile, 'w', 'n', 'UTF-8');
    fwrite(fid, text, 'char');
    fclose(fid);
end

function savePath = createSavePath(folder, config)
    % 创建训练数据保存路径
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    saveRoot = fullfile(pwd, 'saved_train_sets');
    
    if config.isRandom
        modeTag = 'RND';
    else
        modeTag = 'SEQ';
    end
    
    if ~exist(saveRoot, 'dir')
        mkdir(saveRoot);
    end
    
    [~, folderName] = fileparts(folder);
    savePath = fullfile(saveRoot, sprintf('%s_train%dKB_%s_%s', ...
                        folderName, config.trainSizeKB, modeTag, timestamp));
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
end

function validateFileSizes(folder, files, requiredBytes)
    % 验证文件大小
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

%% ========== 翻译效果最差模型对查找 ==========
function [worst2_authors, worst2_V, worst2_table] = findWorst2Models(trainingData, testData, algorithm)
    % 找出混淆矩阵 V 值最大的 2 个人（翻译效果最差的 2 个）
    % V 越大 = 风格差异越大 = 翻译效果越差
    
    authors = {trainingData.author};
    n = length(authors);
    
    maxV = -Inf;
    best_i = 0;
    best_j = 0;
    best_table = [];
    
    fprintf('\n尝试所有 2 人组合（找 V 最大的 - 翻译效果最差的）:\n');
    fprintf('%-15s | %-15s | V值\n', '人员1', '人员2');
    fprintf(repmat('-', 1, 45));
    fprintf('\n');
    
    for i = 1:n-1
        for j = i+1:n
            % 只用这 2 个人做分类
            keep_idx = [i, j];
            trainingData_pair = trainingData(keep_idx);
            testData_pair = testData(ismember({testData.author}, {authors{keep_idx}}));
            
            % 执行 2 人分类
            resultTable = performClassification(trainingData_pair, testData_pair, algorithm);
            v = calculateCramersV(resultTable);
            
            fprintf('%-15s | %-15s | %.4f', authors{i}, authors{j}, v);
            
            % 记录最大的 V（翻译效果最差）
            if v > maxV
                maxV = v;
                best_i = i;
                best_j = j;
                best_table = resultTable;
                fprintf(' ← 目前最大\n');
            else
                fprintf('\n');
            end
        end
    end
    
    fprintf('\n✓ 翻译效果最差的 2 个模型（V最大）: %s 和 %s\n', authors{best_i}, authors{best_j});
    fprintf('  混淆矩阵 V 值: %.4f (风格差异最大，翻译效果最差)\n\n', maxV);
    
    worst2_authors = {authors{best_i}, authors{best_j}};
    worst2_V = maxV;
    worst2_table = best_table;
end

%% ========== 分类执行 ==========
function resultTable = performClassification(trainingData, testData, algorithm)
    % 独立的分类执行模块
    authors = {trainingData.author};
    numAuthors = length(authors);
    resultTable = zeros(numAuthors);
    authorMap = containers.Map(authors, 1:numAuthors);
    
    fprintf('开始分类处理...\n');
    
    for i = 1:length(testData)
        fprintf('处理测试作者 [%d/%d]: %s\n', i, length(testData), testData(i).author);
        
        % 获取真实作者索引
        if authorMap.isKey(testData(i).author)
            trueIdx = authorMap(testData(i).author);
        else
            warning('测试作者 %s 不在训练集中，跳过', testData(i).author);
            continue;
        end
        
        slices = testData(i).slices;
        predictions = zeros(1, length(slices));
        
        for j = 1:length(slices)
            try
                predAuthor = classifySingleSlice(trainingData, slices{j}, algorithm);
                if authorMap.isKey(predAuthor)
                    predictions(j) = authorMap(predAuthor);
                else
                    predictions(j) = randi(numAuthors);
                end
            catch ME
                warning('切片 %d 处理失败: %s', j, ME.message);
                predictions(j) = randi(numAuthors); % 随机猜测
            end
        end
        
        % 统计投票结果
        for j = 1:length(predictions)
            resultTable(trueIdx, predictions(j)) = resultTable(trueIdx, predictions(j)) + 1;
        end
        
        accuracy = sum(predictions == trueIdx) / length(slices) * 100;
        fprintf('  完成 %d 个切片，准确率: %.1f%%\n', length(slices), accuracy);
    end
    
    % 打印报告
    printClassificationReport(resultTable, authors);
end

function bestAuthor = classifySingleSlice(trainingData, testSlice, algorithm)
    % 分类单个切片
    bestAuthor = '';
    minDiff = Inf;
    
    for i = 1:length(trainingData)
        diff = calculateCompressionDiff(trainingData(i).text, testSlice, algorithm);
        if diff < minDiff
            minDiff = diff;
            bestAuthor = trainingData(i).author;
        end
    end
    
    if isempty(bestAuthor)
        error('无法找到最佳匹配作者');
    end
end

function diff = calculateCompressionDiff(trainText, testSlice, algorithm)
    % 计算压缩差异
    combined = [trainText, testSlice];
    combinedSize = compressData(combined, algorithm);
    trainSize = compressData(trainText, algorithm);
    diff = combinedSize - trainSize;
end

%% ========== 压缩算法实现 ==========
function verifyCompressionTools(algorithm)
    if algorithm == 1
        fprintf('✓ GZIP (MATLAB内置) 可用\n');
    elseif algorithm == 2 || algorithm == 6
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
    elseif algorithm == 3
        if ispc
            [status, ~] = system('where bzip2 > nul 2>&1');
        else
            [status, ~] = system('which bzip2 > /dev/null 2>&1');
        end
        if status ~= 0
            error('bzip2未安装!');
        end
        fprintf('✓ BZIP2可用\n');
    elseif algorithm == 4
        if ispc
            [status, ~] = system('where lz4 > nul 2>&1');
        else
            [status, ~] = system('which lz4 > /dev/null 2>&1');
        end
        if status ~= 0
            error('lz4未安装!');
        end
        fprintf('✓ LZ4可用\n');
    elseif algorithm == 5
        if ispc
            [status, ~] = system('where zstd > nul 2>&1');
        else
            [status, ~] = system('which zstd > /dev/null 2>&1');
        end
        if status ~= 0
            error('zstd未安装!');
        end
        fprintf('✓ ZSTD可用\n');
    end
end

function compressedSize = compressData(data, algorithm)
    if algorithm == 1
        compressedSize = compressWithMATLABGzip(data);
    elseif algorithm == 2
        compressedSize = compressWith7Zip(data, 'lzma');
    elseif algorithm == 3
        compressedSize = compressWithBzip2(data);
    elseif algorithm == 4
        compressedSize = compressWithLZ4(data);
    elseif algorithm == 5
        compressedSize = compressWithZstd(data);
    elseif algorithm == 6
        compressedSize = compressWithPPMd(data);
    else
        error('不支持的压缩算法: %d', algorithm);
    end
end

function compressedSize = compressWithMATLABGzip(data)
    tempFile = [tempname '.tmp'];
    gzFile = [tempFile '.gz'];
    
    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, gzFile));
    
    fid = fopen(tempFile, 'wb');
    if fid == -1
        error('无法创建临时文件');
    end
    fwrite(fid, data, 'uint8');
    fclose(fid);
    
    try
        gzip(tempFile, fileparts(tempFile));
        
        maxWait = 5;
        waitInt = 0.1;
        elapsedTime = 0;
        while ~exist(gzFile, 'file') && elapsedTime < maxWait
            pause(waitInt);
            elapsedTime = elapsedTime + waitInt;
        end
        
        if ~exist(gzFile, 'file')
            error('GZIP压缩失败');
        end
        
        dirInfo = dir(gzFile);
        compressedSize = dirInfo.bytes;
        
    catch ME
        rethrow(ME);
    end
end

function compressedSize = compressWith7Zip(data, method)
    tempFile = [tempname '.tmp'];
    if strcmp(method, 'lzma')
        compressedFile = [tempFile '.7z'];
    else
        compressedFile = [tempFile '.ppm7z'];
    end
    
    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, compressedFile));
    
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
    
    try
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
        
    catch ME
        rethrow(ME);
    end
end

function compressedSize = compressWithBzip2(data)
    tempFile = [tempname '.tmp'];
    compressedFile = [tempFile '.bz2'];
    
    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, compressedFile));
    
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
    
    try
        cmd = sprintf('bzip2 -c "%s" > "%s" %s', tempFile, compressedFile, nullRedir);
        status = system(cmd);
        if status ~= 0
            error('bzip2压缩失败');
        end
        
        maxWait = 10;
        waitInt = 0.1;
        elapsedTime = 0;
        while ~exist(compressedFile, 'file') && elapsedTime < maxWait
            pause(waitInt);
            elapsedTime = elapsedTime + waitInt;
        end
        
        if ~exist(compressedFile, 'file')
            error('bzip2压缩失败');
        end
        
        dirInfo = dir(compressedFile);
        compressedSize = dirInfo.bytes;
        
    catch ME
        rethrow(ME);
    end
end

function compressedSize = compressWithLZ4(data)
    tempFile = [tempname '.tmp'];
    compressedFile = [tempFile '.lz4'];
    
    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, compressedFile));
    
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
    
    try
        cmd = sprintf('lz4 -f -q "%s" "%s" %s', tempFile, compressedFile, nullRedir);
        status = system(cmd);
        if status ~= 0
            error('lz4压缩失败');
        end
        
        maxWait = 10;
        waitInt = 0.1;
        elapsedTime = 0;
        while ~exist(compressedFile, 'file') && elapsedTime < maxWait
            pause(waitInt);
            elapsedTime = elapsedTime + waitInt;
        end
        
        if ~exist(compressedFile, 'file')
            error('lz4压缩失败');
        end
        
        dirInfo = dir(compressedFile);
        compressedSize = dirInfo.bytes;
        
    catch ME
        rethrow(ME);
    end
end

function compressedSize = compressWithZstd(data)
    tempFile = [tempname '.tmp'];
    compressedFile = [tempFile '.zst'];
    
    cleanupObj = onCleanup(@() cleanupTempFiles(tempFile, compressedFile));
    
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
    
    try
        cmd = sprintf('zstd -q "%s" -o "%s" %s', tempFile, compressedFile, nullRedir);
        status = system(cmd);
        if status ~= 0
            error('zstd压缩失败');
        end
        
        maxWait = 10;
        waitInt = 0.1;
        elapsedTime = 0;
        while ~exist(compressedFile, 'file') && elapsedTime < maxWait
            pause(waitInt);
            elapsedTime = elapsedTime + waitInt;
        end
        
        if ~exist(compressedFile, 'file')
            error('zstd压缩失败');
        end
        
        dirInfo = dir(compressedFile);
        compressedSize = dirInfo.bytes;
        
    catch ME
        rethrow(ME);
    end
end

function compressedSize = compressWithPPMd(data)
    compressedSize = compressWith7Zip(data, 'ppmd');
end

function cleanupTempFiles(tempFile, compressedFile)
    if exist(tempFile, 'file')
        try delete(tempFile); catch; end
    end
    if exist(compressedFile, 'file')
        try delete(compressedFile); catch; end
    end
end

function cleanupTempFolder()
    tempDir = tempdir;
    tempFiles = dir(fullfile(tempDir, 'tp*.tmp*'));
    for i = 1:length(tempFiles)
        try delete(fullfile(tempDir, tempFiles(i).name)); catch; end
    end
end

%% ========== 结果输出 ==========
function printClassificationReport(table, authors)
    fprintf('\n=== 分类结果矩阵 ===\n');
    
    % 打印列标题
    fprintf('%15s', ' ');
    for i = 1:length(authors)
        fprintf('%12s', authors{i});
    end
    fprintf('\n');
    
    % 打印矩阵
    for i = 1:length(authors)
        fprintf('%12s: ', authors{i});
        for j = 1:length(authors)
            fprintf('%8d', table(i,j));
        end
        fprintf('\n');
    end
    
    total = sum(table(:));
    correct = sum(diag(table));
    fprintf('\n准确率: %.2f%% (%d/%d)\n', correct/total*100, correct, total);
    
    cramerV = calculateCramersV(table);
    fprintf('Cramer''s V系数: %.4f\n', cramerV);
    interpretCramersV(cramerV);
end

function cramerV = calculateCramersV(contingencyTable)
    [chi2, ~, ~] = computeChiSquareFromTable(contingencyTable);
    n = sum(contingencyTable(:));
    k = min(size(contingencyTable));
    if n == 0 || k == 1
        cramerV = 0;
    else
        cramerV = sqrt(chi2 / (n * (k - 1)));
    end
end

function [chi2, p, df] = computeChiSquareFromTable(observed)
    rowTotals = sum(observed, 2);
    colTotals = sum(observed, 1);
    n = sum(observed(:));
    
    if n == 0
        chi2 = 0;
        p = 1;
        df = 0;
        return;
    end
    
    expected = (rowTotals * colTotals) / n;
    expected(expected == 0) = eps;
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
        fprintf('  %.4f - 无关联或极弱关联（翻译效果最好）\n', v);
    elseif v < 0.3
        fprintf('  %.4f - 弱关联（翻译效果较好）\n', v);
    elseif v < 0.5
        fprintf('  %.4f - 中等关联（翻译效果中等）\n', v);
    else
        fprintf('  %.4f - 强关联（翻译效果较差）\n', v);
    end
end

function saveResults(resultTable, trainFolder, config, authors)
    % 保存结果到文件，使用真实的作者名
    [~, folderName] = fileparts(trainFolder);
    algoNames = {'GZIP', 'LZMA', 'BZIP2', 'LZ4', 'ZSTD', 'PPMd'};
    
    excelName = sprintf('Results_%s_%s_T%dKB_S%dKB_N%d.xlsx', ...
                      folderName, algoNames{config.algorithm}, ...
                      config.trainSizeKB, config.testSliceSizeKB, config.testSliceCount);
    
    % 准备Excel数据
    resultCell = cell(size(resultTable, 1) + 1, size(resultTable, 2) + 1);
    resultCell{1,1} = 'True\\Pred';
    for i = 1:length(authors)
        resultCell{1, i+1} = authors{i};
        resultCell{i+1, 1} = authors{i};
        for j = 1:length(authors)
            resultCell{i+1, j+1} = resultTable(i,j);
        end
    end
    
    % 保存Excel
    try
        writecell(resultCell, excelName);
        fprintf('Excel文件已保存: %s\n', excelName);
    catch
        % 如果writecell不可用，使用xlswrite
        xlswrite(excelName, resultCell);
        fprintf('Excel文件已保存(使用xlswrite): %s\n', excelName);
    end
    
    % 保存统计报告
    accuracy = sum(diag(resultTable)) / sum(resultTable(:)) * 100;
    cramerV = calculateCramersV(resultTable);
    [~, pVal, ~] = computeChiSquareFromTable(resultTable);
    
    statsFile = strrep(excelName, '.xlsx', '_stats.txt');
    fid = fopen(statsFile, 'w');
    fprintf(fid, '=== 统计报告 ===\n');
    fprintf(fid, '生成时间: %s\n', datestr(now));
    fprintf(fid, '压缩算法: %s\n', algoNames{config.algorithm});
    fprintf(fid, '训练集大小: %d KB\n', config.trainSizeKB);
    fprintf(fid, '测试切片大小: %d KB\n', config.testSliceSizeKB);
    fprintf(fid, '每个作者切片数: %d\n', config.testSliceCount);
    fprintf(fid, '截取方式: %s\n', iff(config.isRandom, '随机', '顺序'));
    fprintf(fid, '准确率: %.2f%%\n', accuracy);
    fprintf(fid, 'Cramer''s V系数: %.4f\n', cramerV);
    fprintf(fid, '卡方检验p值: %.4e\n', pVal);
    fprintf(fid, '\n=== 混淆矩阵 ===\n');
    fprintf(fid, '行=真实作者，列=预测作者\n\n');
    % 写入作者名作为表头
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
    fclose(fid);
    
    fprintf('统计报告已保存: %s\n', statsFile);
    fprintf('\n结果已保存至当前目录\n');
end

function result = iff(condition, trueVal, falseVal)
    % 简单的三元运算符替代
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end