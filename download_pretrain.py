from tools import pretrain_helper

pretrains = pretrain_helper.fetch_pretrains()

# im eepy today, could be done better -- mikus
def main():
    ind = 0
    for index, i in enumerate(pretrains):
        print(index, i)
        print('=' *50)
        ind = index + 1

    print(ind, 'Download All')

    print('Choose which pretrains you want (number):')
    pretrain_choice = input('> ')
    if int(pretrain_choice) != ind:
        pretrain_helper.download_pretrain(pretrains[int(pretrain_choice)])
        for i in pretrains:
            pretrain_helper.download_pretrain(i)

    elif int(pretrain_choice) == ind:
        for i in pretrains:
            pretrain_helper.download_pretrain(i)
try:
    main()
except KeyboardInterrupt:
    quit()
except:
    main()