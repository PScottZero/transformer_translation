import json

accuracy_and_bleu = json.loads(open('accuracy_and_bleu.json').read())
accuracy_and_bleu = list(accuracy_and_bleu.items())

accuracy_sort = sorted(accuracy_and_bleu, key=lambda x: x[1]['accuracy'], reverse=True)
bleu_sort = sorted(accuracy_and_bleu, key=lambda x: x[1]['bleu'], reverse=True)

with open('rankings.txt', 'w') as f:
  f.write('======== Accuracy Rankings ========\n')
  for model, stats in accuracy_sort:
    f.write(f'{model} {stats["accuracy"]:.3f}\n')

  f.write('\n======== BLEU Rankings ========\n')
  for model, stats in bleu_sort:
    f.write(f'{model} {stats["bleu"]:.3f}\n')
