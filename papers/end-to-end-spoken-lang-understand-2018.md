## Towards end-to-end spoken language understanding

SLU: spoken language understanding, this is different from **Speech Recognition**


### 要点

1. traditional system, pipeline contains:
	* automatic speech recognizer(ASR)
	* classify the text to structured data as domain
	* NLU to determine the intent and extract slots
2. each component(of traditional approach) is trained independently with different optimization objective
	* ASR: minimize the WER(Word Error Rate)
3. End to End SLU(see image below)
4. Discussion
	* not show superior performance
	* With significantly less parameters our system is able to reach 10% relatively worse accuracy
	* to encompass the slot filling task into this framework
	* to explore different architectures for the decoder, such as using different pooling strategies, using deeper networks and incorporating convolutional transformations.
	


![Traditional SLU](/images/slu-old.png)

Traditional SLU system

![end to end SLU](/images/slu-e2e.png)

End to End SLU system(author's proposal)

### 个人点评

1. end to end这个已经提了很久了，特别是当DL概念开始火起来的时候，之前需要pipeline的现在都可以尝试去end to end，所以本文的提法虽然是所谓首创，但也不新鲜
2. 虽然end to end很好理解，但是具体的model的设计和具体的实现，还是有很大差别的，这也是本文作者给出的model的意义，至少不是说model的结构不重要，而事实上在实践中这块往往是最重要的（例如我们经常听人说NN的深度越高越好，但是如何在实际中可训练的基础上来增加深度并不是那么容易，例如gradient vanishing/explosion等问题）
3. 作者提到用到了inhouse的dataset，这就让这个文章的价值大打折扣，让读者无法尝试重现，让peers也无法有相同的metric来评价结果
4. Yoshua Bengio又挂名了，不过不知道具体参加了多少
