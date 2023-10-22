from simple_BERT import detox


if __name__ == '__main__':
    sentences = ["What the hell is going on? I am very confused and pissed off!",
                 "I don't give a fuck.",
                 "I told you she is a bitch.",
                 "This guy is a dick!",
                 "This situation is literally fucked.",
                 "Stop shit-talking, you stupid motherfucker!",
                 "There is only one word to describe this - fuck...",
                 "Damn! It's fucking great!",
                 "Are you fucking kidding me?"]
    print('Censored phrases:')
    print('\n'.join(detox(sentences, True)))
    print('Detoxified phrases:')
    print('\n'.join(detox(sentences)))
