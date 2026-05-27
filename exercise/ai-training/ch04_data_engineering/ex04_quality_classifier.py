"""
练习 4（进阶）：构建训练数据质量分类器
======================================

训练一个 fastText 分类器来区分"高质量"和"低质量"文本。

你需要完成:
1. 准备标注数据:
   - 正例（高质量）: 维基百科段落
   - 负例（低质量）: CommonCrawl 随机采样、自动生成文本
2. 提取文本特征（或使用 fastText 的自动 n-gram 特征）
3. 训练 fastText 分类器
4. 在测试集上评估: 精确率、召回率、F1
5. 用训练好的分类器对新数据进行质量评分

思考题:
- fastText 为什么适合这个任务？和 BERT 分类器相比有什么优劣？
- 如果正负例比例严重不均衡（如 1:100），应该怎么处理？
- 质量分类器的评分可以作为二值过滤（good/bad）还是连续值（0-1 分）？
"""

import fasttext
import random
import os


# TODO: 准备标注数据
def prepare_training_data(positive_texts, negative_texts, output_file):
    """
    TODO: 将标注数据转换为 fastText 格式。

    fastText 格式:
        __label__high_quality This is a good text.
        __label__low_quality This is a bad text.

    参数:
        positive_texts: 高质量文本列表
        negative_texts: 低质量文本列表
        output_file: 输出文件路径
    """
    pass


# TODO: 训练分类器
def train_classifier(train_file, valid_file=None):
    """
    TODO: 训练 fastText 文本分类器。

    使用参数:
        model = fasttext.train_supervised(
            input=train_file,
            lr=1.0,
            epoch=25,
            wordNgrams=2,
            dim=100,
        )
    """
    pass


# TODO: 评估
def evaluate_classifier(model, test_file):
    """
    TODO: 在测试集上评估分类器。
    打印精确率、召回率、F1。
    """
    pass


# TODO: 质量评分
def score_documents(model, texts):
    """
    TODO: 对文档进行质量评分。

    fastText 的 predict 方法可以返回概率:
        labels, probs = model.predict(text, k=2)
    """
    pass


if __name__ == "__main__":
    pass
