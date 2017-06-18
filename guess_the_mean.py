import keras
import numpy as np


score_player = [0,0]
score_machine = [0,0]

print(80*'#')
print("""

This is how the game is played:
1. You will see the sequence of 10 numbers sampled from a normal distribution.
2. Type in your estimate of the mean and press enter.
3. You win a point if you guessed it better than an ANN.

MACHINE VS. HUMAN version 1.0
GOOD LUCK!

""")
print(80*'#')


ai_model = keras.models.load_model('./gauss_player')


while True:

    mean = np.random.uniform(-20,20)
    std = np.random.uniform(0,10)
    seq = np.random.normal(mean, std, (10))
    seq = np.around(seq, 2)
    print('-'*80)

    print('This is the sequence: ', " | ".join(seq.astype(np.str)))
    estimate = float(input('Your estimate? '))
    machine_estimate = ai_model.predict(seq.reshape(1,10,1))[-1][-1][0]
    print('Machine estimate: ', machine_estimate)

    print('Correct answer: ', mean)
    human_error = np.abs(estimate-mean)
    machine_error =  np.abs(machine_estimate-mean)

    score_player[1] += human_error
    score_machine[1] +=  machine_error

    if human_error < machine_error:
        score_player[0]+=1
    else:
        score_machine[0] += 1

    print('Human score: ', score_player[0], 'Human error: ', score_player[1] )

    print('Machine score: ', score_machine[0], 'Machine error: ', score_machine[1])


    print('-'*80)





