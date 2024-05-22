import matplotlib.pyplot as plt


def last_results(results, is_sas=False, print_out=False):
       res = ''
       if not is_sas:
          res += ('Rewards:\n @5 : ' + str(results[-1][5]['reward']) +
             ' @10 : ' + str(results[-1][10]['reward']) +
             ' @15 : ' + str(results[-1][15]['reward']) + ' @20 : ' + str(results[-1][20]['reward']))
          res += '\n\n'
       res += ('Click HR:\n @5 : ' + str(results[-1][5]['click hr']) +
             ' @10 : ' + str(results[-1][10]['click hr']) +
             ' @15 : ' + str(results[-1][15]['click hr']) + ' @20 : ' + str(results[-1][20]['click hr']))
       res += '\n\n'
       res += ('Click NDCG:\n @5 : ' + str(results[-1][5]['click ndcg']) +
             ' @10 : ' + str(results[-1][10]['click ndcg']) +
             ' @15 : ' + str(results[-1][15]['click ndcg']) + ' @20 : ' + str(results[-1][20]['click ndcg']))
       res += '\n\n'
       res += ('Purchase HR:\n @5 : ' + str(results[-1][5]['purchase hr']) +
             ' @10 : ' + str(results[-1][10]['purchase hr']) +
             ' @15 : ' + str(results[-1][15]['purchase hr']) + ' @20 : ' + str(results[-1][20]['purchase hr']))
       res += '\n\n'
       res += ('Purchase NDCG:\n @5 : ' + str(results[-1][5]['purchase ndcg']) +
             ' @10 : ' + str(results[-1][10]['purchase ndcg']) +
             ' @15 : ' + str(results[-1][15]['purchase ndcg']) + ' @20 : ' + str(results[-1][20]['purchase ndcg']))
       if print_out:
            print(res)

       return res



def plot_validation(losses, results, at=20):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))

    rew = [el[at]['reward'] for el in results]
    click_hr = [el[at]['click hr'] for el in results]
    click_ndcg = [el[at]['click ndcg'] for el in results]
    purchase_hr = [el[at]['purchase hr'] for el in results]
    purchase_ndcg = [el[at]['purchase ndcg'] for el in results]

    axes[0, 0].plot(rew)
    axes[0, 0].set_title(f'Rewards @ {at}')

    axes[0, 1].plot(click_hr)
    axes[0, 1].set_title(f'Click HR @ {at}')

    axes[0, 2].plot(click_ndcg)
    axes[0, 2].set_title(f'Click NDCG @ {at}')
    
    axes[1, 0].plot(losses)
    axes[1, 0].set_title('Losses')

    axes[1, 1].plot(purchase_hr)
    axes[1, 1].set_title(f'Purchase HR @ {at}')
    
    axes[1, 2].plot(purchase_ndcg)
    axes[1, 2].set_title(f'Purchase NDCG @ {at}')

    plt.tight_layout()

    plt.show()