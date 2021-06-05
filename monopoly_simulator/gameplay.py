import shutil
import sys
# sys.path.insert(0, '/Users/mwadea/Downloads/GNOME-p3-master-2/')

from monopoly_simulator import initialize_game_elements, background_agent_v4, background_agent_v1_2, \
    background_agent_random, background_agent_v5_1
from monopoly_simulator.action_choices import roll_die
import numpy as np
from monopoly_simulator import card_utility_actions
from monopoly_simulator import background_agent_v3_2, background_agent_v3_1
from monopoly_simulator import read_write_current_state
import json
from monopoly_simulator import novelty_generator
from monopoly_simulator import diagnostics
from monopoly_simulator.agent import Agent
# import xlsxwriter
from monopoly_simulator.flag_config import flag_config_dict
from monopoly_simulator.logging_info import log_file_create
import os
import time
import logging
from datetime import datetime
from monopoly_simulator.rl_agent import RLAgent
from monopoly_simulator import rl_agent_helper
import torch
import torch.nn.functional as F
import csv
import operator

logger = logging.getLogger('monopoly_simulator.logging_info')


def write_history_to_file(game_board, workbook):
    worksheet = workbook.add_worksheet()
    col = 0
    for key in game_board['history']:
        if key == 'param':
            col += 1
            row = 0
            worksheet.write(row, col, key)
            worksheet.write(row, col + 1, 'current_player')
            for item in game_board['history'][key]:
                worksheet.write(row + 1, col, str(item))
                try:
                    worksheet.write(row + 1, col + 1, item['player'].player_name)
                except:
                    pass
                row += 1
            col += 1
        else:
            col += 1
            row = 0
            worksheet.write(row, col, key)
            for item in game_board['history'][key]:
                worksheet.write(row + 1, col, str(item))
                row += 1
    workbook.close()
    print("History logged into history_log.xlsx file.")


def disable_history(game_elements):
    game_elements['history'] = dict()
    game_elements['history']['function'] = list()
    game_elements['history']['param'] = list()
    game_elements['history']['return'] = list()


def simulate_game_instance(game_elements, history_log_file=None, np_seed=2):
    """
    Simulate a game instance.
    :param game_elements: The dict output by set_up_board
    :param np_seed: The numpy seed to use to control randomness.
    :return: None
    """
    logger.debug("size of board " + str(len(game_elements['location_sequence'])))
    np.random.seed(np_seed)
    np.random.shuffle(game_elements['players'])
    game_elements['seed'] = np_seed
    game_elements['card_seed'] = np_seed
    game_elements['choice_function'] = np.random.choice
    count_json = 0   # a counter to keep track of how many rounds the game has to be played before storing the current_state of gameboard to file.
    num_die_rolls = 0
    tot_time = 0
    # game_elements['go_increment'] = 100 # we should not be modifying this here. It is only for testing purposes.
    # One reason to modify go_increment is if your decision agent is not aggressively trying to monopolize. Since go_increment
    # by default is 200 it can lead to runaway cash increases for simple agents like ours.

    logger.debug(
        'players will play in the following order: ' + '->'.join([p.player_name for p in game_elements['players']]))
    logger.debug('Beginning play. Rolling first die...')
    current_player_index = 0
    num_active_players = 4
    winner = None
    workbook = None
    # if history_log_file:
    #     workbook = xlsxwriter.Workbook(history_log_file)
    game_elements['start_time'] = time.time()
    while num_active_players > 1:
        disable_history(
            game_elements)  # comment this out when you want history to stay. Currently, it has high memory consumption, we are working to solve the problem (most likely due to deep copy run-off).
        current_player = game_elements['players'][current_player_index]
        while current_player.status == 'lost':
            current_player_index += 1
            current_player_index = current_player_index % len(game_elements['players'])
            current_player = game_elements['players'][current_player_index]
        current_player.status = 'current_move'

        # pre-roll for current player + out-of-turn moves for everybody else,
        # till we get num_active_players skip turns in a row.

        skip_turn = 0
        if current_player.make_pre_roll_moves(game_elements) == 2:  # 2 is the special skip-turn code
            skip_turn += 1
        out_of_turn_player_index = current_player_index + 1
        out_of_turn_count = 0
        while skip_turn != num_active_players and out_of_turn_count <= 5:  ##oot count reduced to 20 from 200 to keep the game short
            out_of_turn_count += 1
            # print('checkpoint 1')
            out_of_turn_player = game_elements['players'][out_of_turn_player_index % len(game_elements['players'])]
            if out_of_turn_player.status == 'lost':
                out_of_turn_player_index += 1
                continue

            oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
            # add to game history
            game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
            params = dict()
            params['self'] = out_of_turn_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(oot_code)

            if oot_code == 2:
                skip_turn += 1
            else:
                skip_turn = 0
            out_of_turn_player_index += 1

        # now we roll the dice and get into the post_roll phase,
        # but only if we're not in jail.
        # but only if we're not in jail.

        logger.debug("Printing cash balance and net worth of each player: ")
        diagnostics.print_player_net_worths_and_cash_bal(game_elements)

        r = roll_die(game_elements['dies'], np.random.choice)
        for i in range(len(r)):
            game_elements['die_sequence'][i].append(r[i])

        # add to game history
        game_elements['history']['function'].append(roll_die)
        params = dict()
        params['die_objects'] = game_elements['dies']
        params['choice'] = np.random.choice
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(r)

        num_die_rolls += 1
        game_elements['current_die_total'] = sum(r)
        logger.debug('dies have come up ' + str(r))
        if not current_player.currently_in_jail:
            check_for_go = True
            game_elements['move_player_after_die_roll'](current_player, sum(r), game_elements, check_for_go)
            # add to game history
            game_elements['history']['function'].append(game_elements['move_player_after_die_roll'])
            params = dict()
            params['player'] = current_player
            params['rel_move'] = sum(r)
            params['current_gameboard'] = game_elements
            params['check_for_go'] = check_for_go
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

            current_player.process_move_consequences(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.process_move_consequences)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

            # post-roll for current player. No out-of-turn moves allowed at this point.
            current_player.make_post_roll_moves(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.make_post_roll_moves)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

        else:
            current_player.currently_in_jail = False  # the player is only allowed to skip one turn (i.e. this one)

        if current_player.current_cash < 0:
            code = current_player.handle_negative_cash_balance(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(code)
            if code == flag_config_dict['failure_code'] or current_player.current_cash < 0:
                current_player.begin_bankruptcy_proceedings(game_elements)
                # add to game history
                game_elements['history']['function'].append(current_player.begin_bankruptcy_proceedings)
                params = dict()
                params['self'] = current_player
                params['current_gameboard'] = game_elements
                game_elements['history']['param'].append(params)
                game_elements['history']['return'].append(None)

                num_active_players -= 1
                diagnostics.print_asset_owners(game_elements)
                diagnostics.print_player_cash_balances(game_elements)

                if num_active_players == 1:
                    for p in game_elements['players']:
                        if p.status != 'lost':
                            winner = p
                            p.status = 'won'
            else:
                current_player.status = 'waiting_for_move'
        else:
            current_player.status = 'waiting_for_move'

        current_player_index = (current_player_index + 1) % len(game_elements['players'])
        tot_time = time.time() - game_elements['start_time']

        if card_utility_actions.check_for_game_termination(game_elements, tot_time):
            # game terminates if check_for_game_termination returns true.
            # We print some diagnostics and return if any player exceeds this.
            diagnostics.print_asset_owners(game_elements)
            diagnostics.print_player_cash_balances(game_elements)
            logger.debug("Game ran for " + str(tot_time) + " seconds.")
            break

        #This is an example of how you may want to write out gameboard state to file.
        #Uncomment the following piece of code to write out the gameboard current_state to file at the "count_json" iteration.
        #All the data from game_elements will be written to a .json file which can be read back to intialize a new game with
        #those gameboard values to start the game from that point onwards.
        '''
        if count_json == 50:
            outfile = '../current_gameboard_state.json'
            oot_code = read_write_current_state.write_out_current_state_to_file(game_elements, outfile)
            if oot_code == 1:
                print("Successfully written gameboard current state to file.")
                logger.debug("Successfully written gameboard current state to file.")
                print("Cash in hand with players when writing gameboard state to file: ")
                for player in game_elements['players']:
                    print(player.player_name, " current cash=", player.current_cash)
            else:
                print("Something went wrong when trying to write gameboard state to file. "
                      "Rest of the game will be played as normal but will not log state to file.")
        '''
        count_json += 1

    logger.debug('Liquid Cash remaining with Bank = ' + str(game_elements['bank'].total_cash_with_bank))

    if workbook:
        write_history_to_file(game_elements, workbook)
    # let's print some numbers
    logger.debug('printing final asset owners: ')
    diagnostics.print_asset_owners(game_elements)
    logger.debug('number of dice rolls: ' + str(num_die_rolls))
    logger.debug('printing final cash balances: ')
    diagnostics.print_player_cash_balances(game_elements)
    logger.debug("printing net worth of each player: ")
    diagnostics.print_player_net_worths(game_elements)
    logger.debug("Game ran for " + str(tot_time) + " seconds.")

    if winner:
        logger.debug('We have a winner: ' + winner.player_name)
        return winner.player_name
    else:
        winner = card_utility_actions.check_for_winner(game_elements)
        if winner is not None:
            logger.debug('We have a winner: ' + winner.player_name)
            return winner.player_name
        else:
            logger.debug('Game has no winner, do not know what went wrong!!!')
            return None     # ideally should never get here


def set_up_board(game_schema_file_path, player_decision_agents):
    game_schema = json.load(open(game_schema_file_path, 'r'))
    return initialize_game_elements.initialize_board(game_schema, player_decision_agents)


def inject_novelty(current_gameboard, novelty_schema=None):
    """
    Function for illustrating how we inject novelty
    ONLY FOR ILLUSTRATIVE PURPOSES
    :param current_gameboard: the current gameboard into which novelty will be injected. This gameboard will be modified
    :param novelty_schema: the novelty schema json, read in from file. It is more useful for running experiments at scale
    rather than in functions like these. For the most part, we advise writing your novelty generation routines, just like
    we do below, and for using the novelty schema for informational purposes (i.e. for making sense of the novelty_generator.py
    file and its functions.
    :return: None
    """

    ###Below are examples of Level 1, Level 2 and Level 3 Novelties
    ###Uncomment only the Level of novelty that needs to run (i.e, either Level1 or Level 2 or Level 3). Do not mix up novelties from different levels.

    '''
    #Level 1 Novelty

    numberDieNovelty = novelty_generator.NumberClassNovelty()
    numberDieNovelty.die_novelty(current_gameboard, 4, die_state_vector=[[1,2,3,4,5],[1,2,3,4],[5,6,7],[2,3,4]])
    
    classDieNovelty = novelty_generator.TypeClassNovelty()
    die_state_distribution_vector = ['uniform','uniform','biased','biased']
    die_type_vector = ['odd_only','even_only','consecutive','consecutive']
    classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)
    
    classCardNovelty = novelty_generator.TypeClassNovelty()
    novel_cc = dict()
    novel_cc["street_repairs"] = "alternate_contingency_function_1"
    novel_chance = dict()
    novel_chance["general_repairs"] = "alternate_contingency_function_1"
    classCardNovelty.card_novelty(current_gameboard, novel_cc, novel_chance)
    '''

    '''
    #Level 2 Novelty

    #The below combination reassigns property groups and individual properties to different colors.
    #On playing the game it is verified that the newly added property to the color group is taken into account for monopolizing a color group,
    # i,e the orchid color group now has Baltic Avenue besides St. Charles Place, States Avenue and Virginia Avenue. The player acquires a monopoly
    # only on the ownership of all the 4 properties in this case.
    
    inanimateNovelty = novelty_generator.InanimateAttributeNovelty()
    inanimateNovelty.map_property_set_to_color(current_gameboard, [current_gameboard['location_objects']['Park Place'], current_gameboard['location_objects']['Boardwalk']], 'Brown')
    inanimateNovelty.map_property_to_color(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 'Orchid')

    #setting new rents for Indiana Avenue
    inanimateNovelty.rent_novelty(current_gameboard['location_objects']['Indiana Avenue'], {'rent': 50, 'rent_1_house': 150})
    '''

    '''
    #Level 3 Novelty

    granularityNovelty = novelty_generator.GranularityRepresentationNovelty()
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 6)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['States Avenue'], 20)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Tennessee Avenue'], 27)

    spatialNovelty = novelty_generator.SpatialRepresentationNovelty()
    spatialNovelty.color_reordering(current_gameboard, ['Boardwalk', 'Park Place'], 'Blue')

    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Park Place'], 52)
    '''


def play_game():
    """
    Use this function if you want to test a single game instance and control lots of things. For experiments, we will directly
    call some of the functions in gameplay from test_harness.py.

    This is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    but still relatively simple agent soon.
    :return: String. the name of the player who won the game, if there was a winner, otherwise None.
    """

    try:
        os.makedirs('../single_tournament/')
        print('Creating folder and logging gameplay.')
    except:
        print('Logging gameplay.')

    logger = log_file_create('../single_tournament/seed_6.log')
    player_decision_agents = dict()
    # for p in ['player_1','player_3']:
    #     player_decision_agents[p] = simple_decision_agent_1.decision_agent_methods

    player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_2'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods, load_path = None)

    player_decision_agents['player_4'] = Agent(**background_agent_v3_1.decision_agent_methods)

    game_elements = set_up_board('monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''

    inject_novelty(game_elements)

    if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
        logger.error("Error in initializing agents. Cannot play the game.")
        return None
    else:
        logger.debug("Sucessfully initialized all player agents.")
        winner = simulate_game_instance(game_elements)
        if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
            logger.error("Error in agent shutdown.")
            handlers_copy = logger.handlers[:]
            for handler in handlers_copy:
                logger.removeHandler(handler)
                handler.close()
                handler.flush()
            return None
        else:
            logger.debug("All player agents have been shutdown. ")
            logger.debug("GAME OVER")
            handlers_copy = logger.handlers[:]
            for handler in handlers_copy:
                logger.removeHandler(handler)
                handler.close()
                handler.flush()
            return winner


def play_game_in_tournament(game_seed, rl_agent, inject_novelty_function=None, against=0, load_path=None,
                            opponent=None, eps=0.1):
    logger.debug('seed used: ' + str(game_seed))
    player_decision_agents = dict()
    # for p in ['player_1','player_3']:
    #     player_decision_agents[p] = simple_decision_agent_1.decision_agent_methods
    if against == -2:
        player_decision_agents['player_1'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)

        # player_decision_agents['player_1'] = rl_agent[1]
        player_decision_agents['player_2'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
        # player_decision_agents['player_2'] = rl_agent[1]
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path,train=False,eps=eps)
        # player_decision_agents['player_4'] = rl_agent[1]
        player_decision_agents['player_4'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
    if against == -1:
        player_decision_agents['player_1'] = Agent(**background_agent_v1_2.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v1_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v1_2.decision_agent_methods)

    if against == 0:
        player_decision_agents['player_1'] = Agent(**background_agent_v1_2.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v1_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 1:
        player_decision_agents['player_1'] = Agent(**background_agent_v1_2.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v3_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 2:
        player_decision_agents['player_1'] = Agent(**background_agent_v3_2.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v3_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 3:
        player_decision_agents['player_1'] = Agent(**background_agent_v5_1.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v5_1.decision_agent_methods)
        player_decision_agents['player_3'] = rl_agent[0]
        # player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
        #                                              load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v5_1.decision_agent_methods)

    elif against == 4:
        player_decision_agents['player_1'] = Agent(**background_agent_v5_1.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v5_1.decision_agent_methods)
        player_decision_agents['player_3'] = rl_agent[0]
        # player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
        #                                              load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 5:
        player_decision_agents['player_1'] = Agent(**background_agent_v5_1.decision_agent_methods)
        player_decision_agents['player_2'] = Agent(**background_agent_v3_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 6:
        # player_decision_agents['player_1'] = rl_agent[1]
        player_decision_agents['player_1'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
        player_decision_agents['player_2'] = Agent(**background_agent_v1_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v1_2.decision_agent_methods)

    elif against == 7:
        # player_decision_agents['player_1'] = rl_agent[1]
        player_decision_agents['player_1'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
        player_decision_agents['player_2'] = Agent(**background_agent_v3_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    elif against == 8:
        # player_decision_agents['player_1'] = rl_agent[1]
        player_decision_agents['player_1'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
        player_decision_agents['player_2'] = Agent(**background_agent_v5_1.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v5_1.decision_agent_methods)

    elif against == 9:
        # player_decision_agents['player_1'] = rl_agent[1]
        player_decision_agents['player_1'] = RLAgent(**background_agent_random.decision_agent_methods,
                                                     load_path=opponent, train=False, eps=0.0)
        player_decision_agents['player_2'] = Agent(**background_agent_v1_2.decision_agent_methods)
        # player_decision_agents['player_3'] = rl_agent[0]
        player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods,
                                                     load_path=load_path, train=False, eps=eps)
        player_decision_agents['player_4'] = Agent(**background_agent_v3_2.decision_agent_methods)

    game_elements = set_up_board('../monopoly_game_schema_v1-2.json',
                                 player_decision_agents)
    
    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''

    if inject_novelty_function:
        inject_novelty_function(game_elements)

    if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
        logger.error("Error in initializing agents. Cannot play the game.")
        return None
    else:
        logger.debug("Sucessfully initialized all player agents.")
        winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
        if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
            logger.error("Error in agent shutdown.")
            return None
        else:
            logger.debug("All player agents have been shutdown. ")
            logger.debug("GAME OVER")
            return winner


def training_games(n_games, load_path):
    """
    similar to play_game, but instead of playing only one game, the player needs to play multiple games in order to
    train the RLAgent.
    :return: The trained RLAgent?
    """
    checkpoint_folder = '../checkpoints/'
    checkpoint_prefix = 'checkpoint_'
    checkpoint_suffix = '_' + datetime.now().strftime('%m%d%H%M') + '.tar'

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    try:
        os.makedirs('../training_tournament/')
        print('Creating folder and logging gameplay.')
    except:
        print('Logging gameplay.')


    avg_wins = dict()
    player_decision_agents = dict()

    # Trevor: Initialize the players once. Test if we can use the same players to play multiple games - We can!!
    player_decision_agents['player_1'] = Agent(**background_agent_v5_1.decision_agent_methods)
    player_decision_agents['player_2'] = Agent(**background_agent_v5_1.decision_agent_methods)
    player_decision_agents['player_3'] = RLAgent(**background_agent_v4.decision_agent_methods, load_path=load_path)
    player_decision_agents['player_4'] = Agent(**background_agent_v5_1.decision_agent_methods)
    wins = {'player_1': 0,
            'player_2': 0,
            'player_3': 0,
            'player_4': 0}
    wins_last_1000 = {'player_1': 0,
                    'player_2': 0,
                    'player_3': 0,
                    'player_4': 0}
    # Trevor: Need to setup game elements for each game - this will have to be in the for loop ?
    for i in range(n_games):  # Trevor: EPISODES FOR LOOP - We can loop over random seeds as done for tournaments.
        print('*' * 30 + '--New Game Instance: Game', str(i) + '--' + '*' * 30)
        logger = log_file_create('../training_tournament/testing.log')
        game_elements = set_up_board(
            '../monopoly_game_schema_v1-2.json',
            player_decision_agents)

        if player_decision_agents['player_1'].startup(game_elements) == -1 or player_decision_agents[
            'player_2'].startup(
            game_elements) == -1 or \
                player_decision_agents['player_3'].startup(game_elements) == -1 or player_decision_agents[
            'player_4'].startup(game_elements) == -1:
            logger.error("Error in initializing agents. Cannot play the game.")
            return None
        else:
            logger.debug("Sucessfully initialized all player agents.")
            winner, _, _ = simulate_game_instance_training(game_elements, np_seed=i)
            if winner is not None:
                wins[winner] += 1
                wins_last_1000[winner] += 1

            if i != 0 and i % 100 == 0:
                print('Win count: ', wins)
                print('Wins last 1000', wins_last_1000)
                avg_wins[i] = wins_last_1000['player_3']
                # print("RL wins: ", avg_wins)
                wins_last_1000 = {'player_1': 0,
                                 'player_2': 0,
                                 'player_3': 0,
                                 'player_4': 0}

                with open("avg_wins_v5.txt", "a", newline="") as Winfile:
                    result_writer = csv.writer(Winfile, delimiter=",")
                    result_writer.writerow([avg_wins[i]])

            # Trevor: LOAD TARGET NET EVERY FIXED NUMBER OF EPISODES
            # if i % player_decision_agents['player_3'].target_update == 0:
            #     player_decision_agents['player_3'].target_net.load_state_dict(player_decision_agents['player_3']
            #                                                                   .policy_net.state_dict())

            # SAVE THE NETWORK PARAMETERS, EXPERIENCE
            if i != 0 and i % 30000 == 0:
                checkpoint_path = checkpoint_folder + \
                                  checkpoint_prefix + \
                                  str(int(i)) + "_r5_s2_v5_" + \
                                  checkpoint_suffix
                torch.save({'episode': i,
                            'model_state_dict': player_decision_agents['player_3'].policy_net.state_dict(),
                            'optimizer_state_dict': player_decision_agents['player_3'].optimizer.state_dict(),
                            'loss': player_decision_agents['player_3'].loss,
                            'replay_memory': player_decision_agents['player_3'].memory,
                            'current_step': player_decision_agents['player_3'].current_step,
                            'state': player_decision_agents['player_3'].state,
                            'action': player_decision_agents['player_3'].action,
                            'reward': player_decision_agents['player_3'].reward,
                            'next_state': player_decision_agents['player_3'].next_state,
                            'current_game_count': player_decision_agents['player_3'].current_game_count,
                            'win_dict': wins,
                            'wins_last_100': wins_last_1000
                            }, checkpoint_path)

            if player_decision_agents['player_1'].shutdown() == -1 or player_decision_agents[
                'player_2'].shutdown() == -1 or \
                    player_decision_agents['player_3'].shutdown() == -1 or player_decision_agents[
                'player_4'].shutdown() == -1:
                logger.error("Error in agent shutdown.")
                handlers_copy = logger.handlers[:]
                for handler in handlers_copy:
                    logger.removeHandler(handler)
                    handler.close()
                    handler.flush()
                return None
            else:
                logger.debug("All player agents have been shutdown. ")
                logger.debug("GAME OVER")
                handlers_copy = logger.handlers[:]
                for handler in handlers_copy:
                    logger.removeHandler(handler)
                    handler.close()
                    handler.flush()
                print(winner, 'won the game')
                # return winner
                if winner == "player_3":
                    print('Number of failed actions this episode',
                          player_decision_agents["player_3"].failed_actions)
                    print('Number of successful actions this episode',
                          player_decision_agents["player_3"].successful_actions)
                    print('Number of actions in memory', len(player_decision_agents["player_3"].memory.memory))
                    print('Exploration Rate', player_decision_agents["player_3"].strategy.get_exploration_rate(
                        player_decision_agents["player_3"].current_step))
                    print('Avg. Episodic Reward',
                          player_decision_agents["player_3"].episodic_rewards / (
                                  player_decision_agents["player_3"].successful_actions + player_decision_agents[
                              "player_3"].failed_actions))
                    player_decision_agents["player_3"].episodic_rewards = 0
                    player_decision_agents["player_3"].episodic_step = 0
                    player_decision_agents["player_3"].failed_actions = 0
                    player_decision_agents["player_3"].successful_actions = 0
                    # Set the state to all zeros so it is ignored in next q value calculations
                    player_decision_agents["player_3"].next_state = torch.zeros(
                        player_decision_agents["player_3"].state.shape).to(
                        player_decision_agents["player_3"].device)
                    player_decision_agents["player_3"].reward = torch.Tensor([1000]).to(
                        player_decision_agents["player_3"].device)
                    player_decision_agents["player_3"].memory.push(
                        rl_agent_helper.Experience(player_decision_agents["player_3"].state,
                                                   player_decision_agents["player_3"].action,
                                                   player_decision_agents["player_3"].next_state,
                                                   player_decision_agents["player_3"].reward))

                else:
                    print('Number of failed actions this episode', player_decision_agents["player_3"].failed_actions)
                    print('Number of successful actions this episode',
                          player_decision_agents["player_3"].successful_actions)
                    print('Number of actions in memory', len(player_decision_agents["player_3"].memory.memory))
                    print('Exploration Rate', player_decision_agents["player_3"].strategy.get_exploration_rate(
                        player_decision_agents["player_3"].current_step))
                    print('Avg. Episodic Reward',
                          player_decision_agents["player_3"].episodic_rewards / (
                                  player_decision_agents["player_3"].successful_actions + player_decision_agents[
                              "player_3"].failed_actions))
                    player_decision_agents["player_3"].episodic_rewards = 0
                    player_decision_agents["player_3"].episodic_step = 0
                    player_decision_agents["player_3"].failed_actions = 0
                    player_decision_agents["player_3"].successful_actions = 0
                    # Set the state to all zeros so it is ignored in next q value calculations
                    player_decision_agents["player_3"].next_state = torch.zeros(player_decision_agents["player_3"].state.shape).to(
                        player_decision_agents["player_3"].device)
                    player_decision_agents["player_3"].reward = torch.Tensor([-2]).to(
                        player_decision_agents["player_3"].device)
                    player_decision_agents["player_3"].memory.push(
                        rl_agent_helper.Experience(player_decision_agents["player_3"].state,
                                                   player_decision_agents["player_3"].action,
                                                   player_decision_agents["player_3"].next_state,
                                                   player_decision_agents["player_3"].reward))



def simulate_game_instance_training(game_elements, history_log_file=None, np_seed=2):
    """
    Similar to simulate_game_instance, but used in training of the RLAgent.
    Trevor Comments for RL Agent Training:
    1. If the player has lost, the move phases will not be called, so will need to handle it here -
        The reward will be -1
    2. If the player has won, even then we need to update the reward here -
        The reward will be 1
    3. Where do we push the (state, action, reward, next_state) onto the memory?
        Currently we push in the move phases (player.py) and when either the player wins or loses here (gameplay.py)
        TODO: Push when player.py returns skip_turn/concluded action

    :param game_elements: The dict output by set_up_board
    :param np_seed: The numpy seed to use to control randomness.
    :return: None
    """
    mean_q = []
    mean_reward = []
    logger.debug("size of board " + str(len(game_elements['location_sequence'])))
    np.random.seed(np_seed)
    np.random.shuffle(game_elements['players'])
    game_elements['seed'] = np_seed
    game_elements['card_seed'] = np_seed
    game_elements['choice_function'] = np.random.choice
    count_json = 0  # a counter to keep track of how many rounds the game has to be played before storing the current_state of gameboard to file.
    num_die_rolls = 0
    tot_time = 0
    # game_elements['go_increment'] = 100 # we should not be modifying this here. It is only for testing purposes.
    # One reason to modify go_increment is if your decision \gent is not aggressively trying to monopolize. Since go_increment
    # by default is 200 it can lead to runaway cash increases for simple agents like ours.

    logger.debug(
        'players will play in the following order: ' + '->'.join([p.player_name for p in game_elements['players']]))
    logger.debug('Beginning play. Rolling first die...')
    current_player_index = 0
    num_active_players = 4
    winner = None
    workbook = None
    # if history_log_file:
    #     workbook = xlsxwriter.Workbook(history_log_file)
    game_elements['start_time'] = time.time()
    while num_active_players > 1:
        disable_history(
            game_elements)  # comment this out when you want history to stay. Currently, it has high memory consumption, we are working to solve the problem (most likely due to deep copy run-off).
        current_player = game_elements['players'][current_player_index]
        if isinstance(current_player.agent, RLAgent) and current_player.status != 'lost':
            if current_player.agent.memory.can_provide_sample(current_player.agent.batch_size):
                experiences = current_player.agent.memory.sample(current_player.agent.batch_size)
                background_agent_v4.learn(current_player.agent.agent_sac,experiences=experiences)


        while current_player.status == 'lost':
            current_player_index += 1
            current_player_index = current_player_index % len(game_elements['players'])
            current_player = game_elements['players'][current_player_index]
        current_player.status = 'current_move'

        # pre-roll for current player + out-of-turn moves for everybody else,
        # till we get num_active_players skip turns in a row.

        skip_turn = 0
        if current_player.make_pre_roll_moves(game_elements) == 2:  # 2 is the special skip-turn code
            skip_turn += 1
        out_of_turn_player_index = current_player_index + 1
        out_of_turn_count = 0
        while skip_turn != num_active_players and out_of_turn_count <= 5:  ##oot count reduced to 20 from 200 to keep the game short
            out_of_turn_count += 1
            # print('checkpoint 1')
            out_of_turn_player = game_elements['players'][out_of_turn_player_index % len(game_elements['players'])]
            if out_of_turn_player.status == 'lost':
                out_of_turn_player_index += 1
                continue

            oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
            # add to game history
            game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
            params = dict()
            params['self'] = out_of_turn_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(oot_code)

            if oot_code == 2:
                skip_turn += 1
            else:
                skip_turn = 0
            out_of_turn_player_index += 1

        # now we roll the dice and get into the post_roll phase,
        # but only if we're not in jail.
        # but only if we're not in jail.

        logger.debug("Printing cash balance and net worth of each player: ")
        diagnostics.print_player_net_worths_and_cash_bal(game_elements)

        r = roll_die(game_elements['dies'], np.random.choice)
        for i in range(len(r)):
            game_elements['die_sequence'][i].append(r[i])

        # add to game history
        game_elements['history']['function'].append(roll_die)
        params = dict()
        params['die_objects'] = game_elements['dies']
        params['choice'] = np.random.choice
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(r)

        num_die_rolls += 1
        game_elements['current_die_total'] = sum(r)
        logger.debug('dies have come up ' + str(r))
        if not current_player.currently_in_jail:
            check_for_go = True
            game_elements['move_player_after_die_roll'](current_player, sum(r), game_elements, check_for_go)
            # add to game history
            game_elements['history']['function'].append(game_elements['move_player_after_die_roll'])
            params = dict()
            params['player'] = current_player
            params['rel_move'] = sum(r)
            params['current_gameboard'] = game_elements
            params['check_for_go'] = check_for_go
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

            current_player.process_move_consequences(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.process_move_consequences)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

            # post-roll for current player. No out-of-turn moves allowed at this point.
            current_player.make_post_roll_moves(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.make_post_roll_moves)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(None)

        else:
            current_player.currently_in_jail = False  # the player is only allowed to skip one turn (i.e. this one)

        if current_player.current_cash < 0:
            code = current_player.handle_negative_cash_balance(game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
            params = dict()
            params['self'] = current_player
            params['current_gameboard'] = game_elements
            game_elements['history']['param'].append(params)
            game_elements['history']['return'].append(code)
            if code == flag_config_dict['failure_code'] or current_player.current_cash < 0:
                print(current_player.player_name, 'lost the game')
                # Trevor: Player LOST!! This is where the episode has ended for a given player!
                # if isinstance(current_player.agent, RLAgent):
                #     print('Number of failed actions this episode', current_player.agent.failed_actions)
                #     print('Number of successful actions this episode', current_player.agent.successful_actions)
                #     print('Number of actions in memory', len(current_player.agent.memory.memory))
                #     print('Exploration Rate', current_player.agent.strategy.get_exploration_rate(
                #         current_player.agent.current_step))
                #     print('Avg. Episodic Reward',
                #           current_player.agent.episodic_rewards / (
                #                   current_player.agent.successful_actions + current_player.agent.failed_actions))
                #     current_player.agent.episodic_rewards = 0
                #     current_player.agent.episodic_step = 0
                #     current_player.agent.failed_actions = 0
                #     current_player.agent.successful_actions = 0
                #     # Set the state to all zeros so it is ignored in next q value calculations
                #     current_player.agent.next_state = torch.zeros(current_player.agent.state.shape).to(
                #         current_player.agent.device)
                #     current_player.agent.reward = torch.Tensor([-2]).to(current_player.agent.device)
                #     current_player.agent.memory.push(rl_agent_helper.Experience(current_player.agent.state,
                #                                                                 current_player.agent.action,
                #                                                                 current_player.agent.next_state,
                #                                                                 current_player.agent.reward))
                current_player.begin_bankruptcy_proceedings(game_elements)
                # add to game history
                game_elements['history']['function'].append(current_player.begin_bankruptcy_proceedings)
                params = dict()
                params['self'] = current_player
                params['current_gameboard'] = game_elements
                game_elements['history']['param'].append(params)
                game_elements['history']['return'].append(None)

                num_active_players -= 1
                diagnostics.print_asset_owners(game_elements)
                diagnostics.print_player_cash_balances(game_elements)

                if num_active_players == 1:
                    for p in game_elements['players']:
                        if p.status != 'lost':
                            winner = p
                            # Trevor: If the RLAgent wins - We need to push rewards and states here.
                            # if isinstance(p.agent, RLAgent):
                            #     print('Number of failed actions this episode', p.agent.failed_actions)
                            #     print('Number of successful actions this episode',
                            #           p.agent.successful_actions)
                            #     print('Number of actions in memory', len(p.agent.memory.memory))
                            #     print('Exploration Rate', p.agent.strategy.get_exploration_rate(
                            #         p.agent.current_step))
                            #     print('Avg. Episodic Reward',
                            #           p.agent.episodic_rewards / (
                            #                   p.agent.successful_actions + p.agent.failed_actions))
                            #     p.agent.episodic_rewards = 0
                            #     p.agent.episodic_step = 0
                            #     p.agent.failed_actions = 0
                            #     p.agent.successful_actions = 0
                            #     # Set the state to all zeros so it is ignored in next q value calculations
                            #     p.agent.next_state = torch.zeros(p.agent.state.shape).to(p.agent.device)
                            #     p.agent.reward = torch.Tensor([1000]).to(p.agent.device)
                            #     p.agent.memory.push(
                            #         rl_agent_helper.Experience(p.agent.state,
                            #                                    p.agent.action,
                            #                                    p.agent.next_state,
                            #                                    p.agent.reward))
                            p.status = 'won'
            else:
                current_player.status = 'waiting_for_move'
        else:
            current_player.status = 'waiting_for_move'

        current_player_index = (current_player_index + 1) % len(game_elements['players'])
        tot_time = time.time() - game_elements['start_time']

        if card_utility_actions.check_for_game_termination(game_elements, tot_time):
            # game terminates if check_for_game_termination returns true.
            # We print some diagnostics and return if any player exceeds this.
            diagnostics.print_asset_owners(game_elements)
            diagnostics.print_player_cash_balances(game_elements)
            logger.debug("Game ran for " + str(tot_time) + " seconds.")
            break

        # This is an example of how you may want to write out gameboard state to file.
        # Uncomment the following piece of code to write out the gameboard current_state to file at the "count_json" iteration.
        # All the data from game_elements will be written to a .json file which can be read back to intialize a new game with
        # those gameboard values to start the game from that point onwards.
        '''
        if count_json == 50:
            outfile = '../current_gameboard_state.json'
            oot_code = read_write_current_state.write_out_current_state_to_file(game_elements, outfile)
            if oot_code == 1:
                print("Successfully written gameboard current state to file.")
                logger.debug("Successfully written gameboard current state to file.")
                print("Cash in hand with players when writing gameboard state to file: ")
                for player in game_elements['players']:
                    print(player.player_name, " current cash=", player.current_cash)
            else:
                print("Something went wrong when trying to write gameboard state to file. "
                      "Rest of the game will be played as normal but will not log state to file.")
        '''
        count_json += 1

    logger.debug('Liquid Cash remaining with Bank = ' + str(game_elements['bank'].total_cash_with_bank))

    if workbook:
        write_history_to_file(game_elements, workbook)
    # let's print some numbers
    logger.debug('printing final asset owners: ')
    diagnostics.print_asset_owners(game_elements)
    logger.debug('number of dice rolls: ' + str(num_die_rolls))
    logger.debug('printing final cash balances: ')
    diagnostics.print_player_cash_balances(game_elements)
    logger.debug("printing net worth of each player: ")
    diagnostics.print_player_net_worths(game_elements)
    logger.debug("Game ran for " + str(tot_time) + " seconds.")
    # if len(mean_q) > 0:
    #     mean_qval = sum(mean_q) / len(mean_q)
    # if len(mean_reward) > 0:
    #     mean_rval = sum(mean_reward) / len(mean_reward)
    mean_qval, mean_rval=0,0
    if winner:
        logger.debug('We have a winner: ' + winner.player_name)
        return winner.player_name, mean_qval, mean_rval
    else:
        winner = card_utility_actions.check_for_winner(game_elements)
        if winner is not None:
            logger.debug('We have a winner: ' + winner.player_name)
            return winner.player_name, mean_qval, mean_rval
        else:
            logger.debug('Game has no winner, do not know what went wrong!!!')
            return None, mean_qval, mean_rval  # ideally should never get here


def play_tournament_without_novelty(tournament_log_folder=None, meta_seed=5, num_games=100, load_path = None,
                                    eps = 0.1, opponent = None):
    """
    Tournament logging is not currently supported, but will be soon.
    :param tournament_log_folder: String. The path to a folder.
    :param meta_seed: This is the seed we will use to generate a sequence of seeds, that will (in turn) spawn the games in gameplay/simulate_game_instance
    :param num_games: The number of games to simulate in a tournament
    :return: None. Will print out the win-loss metrics, and will write out game logs
    """

    if not tournament_log_folder:
        print("No logging folder specified, cannot log tournaments. Provide a logging folder path.")
        raise Exception

    np.random.seed(meta_seed)
    big_list = list(range(0, 1000000))
    # np.random.shuffle(big_list)
    tournament_seeds = big_list[0:num_games]
    winners = list()
    count = 1
    # wins = {'player_1': 0,
    #         'player_2': 0,
    #         'player_3': 0,
    #         'player_4': 0}
    folder_name = "../tournament_logs" + tournament_log_folder
    try:
        os.makedirs(folder_name)
        # print('Logging gameplay')
    except:
        # print('Given logging folder already exists. Clearing folder before logging new files.')
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)

    metadata_dict = {
        "function": "play_tournament_without_novelty",
        "parameters": {
            "meta_seed": meta_seed,
            "num_game": num_games
        }
    }

    json_filename = folder_name + "tournament_meta_data.json"
    out_file = open(json_filename, "w")
    json.dump(metadata_dict, out_file, indent=4)
    out_file.close()
    print('Loading RL Agent')
    rl_agent = [RLAgent(**background_agent_v4.decision_agent_methods,
                        load_path=load_path,
                        train=False,
                        eps=eps
                        ), RLAgent(**background_agent_random.decision_agent_methods,
                                   load_path=opponent,
                                   train=False,
                                   eps=0.0
                                   )]
    # print('RL Agent loaded')
    # for against in [-1, 0, 1, 2, 3]:
    wins_dict = dict()
    tour_win = dict()
    # tour_win = {'player_1': 0,
    #         'player_2': 0,
    #         'player_3': 0,
    #         'player_4': 0}
    # for against in [1,2,0,-1,3,4,5,-2]:
    # for against in [7, 8]:
    # for against in [-2, 3, 4, 5, 1, 2, 0, 7, 8]:
    for against in [3, 4, 2, 1, 5, 0]:
        wins = {'player_1': 0,
                'player_2': 0,
                'player_3': 0,
                'player_4': 0}
        for t in tournament_seeds:
            # print('Logging gameplay for seed: ', str(t), ' ---> Game ' + str(count))
            filename = folder_name + "meta_seed_" + str(meta_seed) + '_num_games_' + str(count) + '.log'
            logger = log_file_create(filename)
            winner = play_game_in_tournament(t, rl_agent, against=against, load_path = load_path, opponent=opponent,
                                             eps = 0.1)
            if winner is not None:
                wins[winner] += 1
            # print(winner)
            handlers_copy = logger.handlers[:]
            for handler in handlers_copy:
                logger.removeHandler(handler)
                handler.close()
                handler.flush()
            count += 1
        # print('Against:', against)
        # print('Wins:', "[A]", against, wins)
        wins_dict[against] = wins["player_3"]
        max_value = max(wins.values())  # maximum value
        max_keys = [k for k, v in wins.items() if v == max_value]
        # winner = max(wins.items(), key=operator.itemgetter(1))
        # print(max_keys)
        if 'player_3' in max_keys:
            winner = 'player_3'
        else:
            winner = max_keys[0]
        # print("Winner: ", winner)
        tour_win[against] = winner
    return wins_dict, tour_win


# training_games(60000, load_path='../checkpoints/checkpoint_0_10111954.tar')
training_games(10000, load_path=None)
