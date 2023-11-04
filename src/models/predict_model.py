from simple_BERT import detox as simple_detox
from cond_BERT import load_condBERT


if __name__ == '__main__':
    sentences = ["What the hell is going on? I am very confused and pissed off!",
                 "I don't give a fuck.",
                 "I told you she is a bitch.",
                 "This guy is a dick!",
                 "This situation is literally fucked.",
                 "Stop shit-talking, you stupid motherfucker!",
                 "There is only one word to describe this - fuck...",
                 "Damn! It's fucking great!",
                 "Are you fucking kidding me?",
                 "She is always bitchy about him!"]
    print('Original:')
    print('\n'.join(sentences))
    print('-' * 150)
    print('simple_BERT:')
    print('\n'.join(simple_detox(sentences)))
    condBERT = load_condBERT()
    print('cond_BERT:')
    print('\n'.join(condBERT.detox(sentences)))
