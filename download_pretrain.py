from tools import pretrain_helper

pretrains = pretrain_helper.fetch_pretrains()

for index, i in enumerate(pretrains):
    print(index, i)
    print('=' *50)

print('Choose which pretrains you want (number):')
pretrain_choice = input('> ')

pretrain_helper.download_pretrain(pretrains[int(pretrain_choice)])
