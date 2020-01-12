import itertools
import time
import operator
import nltk
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from collections import defaultdict
from nltk.corpus import wordnet as wn

'''
Number of applicants: 7428
Number of different skills: 4480

An applicant 'a' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})

Likewise, a project 'p' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})
'''

nltk.download('wordnet')
set_students_dblp = set()
set_students_imdb = set()
distances_dblp = {}
distances_imdb = {}
skills_dblp = []
skills_imdb = []


def prepare_data_dblp():
    # Read skills file and create applicants tuples in set_A
    global set_students_dblp
    with open("data-dblp/DBLP_skill.csv", 'r', encoding='latin1') as temp_f:
        lines = temp_f.readlines()
        for index, line in enumerate(lines):
            split_line = line.rstrip('\n').split(',')
            if len(split_line) > 1:
                set_students_dblp.add((index, frozenset(split_line[1:])))
    temp_f.close()

    # Gets frequency of terms
    freq_skills = defaultdict(int)
    for applicant in set_students_dblp:
        applicant_skills = applicant[1]
        for skill in applicant_skills:
            freq_skills[skill] += 1

    global distances_dblp
    try:
        print('Found DBLP "distances.dat" pre-computed term similarities file.')
        distances_dblp = pickle.load(open("data-dblp/distances.dat", "rb"))
    except (OSError, IOError) as e:
        # Pre compute all shortest distances between skills in the tree
        print('Not found DBLP "distances.dat" pre-computed term similarities file. Calculating...')
        for term1 in freq_skills.keys():
            print('\tCalculating similarity for term "{}"'.format(term1))
            for term2 in freq_skills.keys():
                try:
                    term1_synset = wn.synset(term1 + '.n.01')
                    term2_synset = wn.synset(term2 + '.n.01')
                    score = term1_synset.path_similarity(term2_synset)
                except nltk.corpus.reader.wordnet.WordNetError:
                    score = 0
                if term1 in distances_dblp:
                    distances_dblp[term1][term2] = score
                else:
                    distances_dblp[term1] = {term2: score}
        print('Done! Saving DBLP "distances.dat" pre-computed term similarities file.')
        pickle.dump(distances_dblp, open("data-dblp/distances.dat", "wb"))

    # Gets 2k most frequent terms from all the 4480 terms
    global skills_dblp
    skills_dblp = sorted(freq_skills.items(), key=lambda item: item[1], reverse=True)[:2000]


def get_genres_from_titles(titles, tid):
    try:
        genres = titles.loc[tid].genres
        if genres == '\\N':
            return ''
        else:
            return genres
    except KeyError:
        return ''


def prepare_data_imdb():
    global directors
    try:
        directors = pickle.load(open("data-imdb/directors.dat", "rb"))
        print('Found IMDB "directors.dat" pre-computed data file.')
    except (OSError, IOError) as e:
        print('Not found IMDB "directors.dat" pre-computed term similarities file. Calculating...')
        # Reads all 9.798.254 names and removes not needed columns - https://datasets.imdbws.com/name.basics.tsv.gz
        namebasics = pd.read_csv('data-imdb/name.basics.tsv', delimiter='\t', encoding='utf-8', low_memory=False)
        del namebasics['birthYear']
        del namebasics['deathYear']
        # Filters to 301.153 directors from names, and removes comma char
        directors = namebasics[namebasics['primaryProfession'].str.match('(?<!_)director', na=False)]
        del directors['primaryProfession']
        directors['primaryName'] = directors['primaryName'].str.replace(',', '')
        # Filters to 274.058 directors who have at least one movie
        directors = directors[directors['knownForTitles'].str.len() > 2]
        directors = directors.set_index('nconst')
        # Converts strings of titles to lists
        directors.knownForTitles = directors.knownForTitles.apply(lambda x: x.split(','))

        # Reads titles file and removes not needed columns - https://datasets.imdbws.com/title.basics.tsv.gz
        titlebasics = pd.read_csv('data-imdb/title.basics.tsv', delimiter='\t', encoding='utf-8', low_memory=False)
        del titlebasics['titleType']
        del titlebasics['originalTitle']
        del titlebasics['isAdult']
        del titlebasics['startYear']
        del titlebasics['endYear']
        del titlebasics['runtimeMinutes']
        # Removes null items
        titlebasics = titlebasics.dropna()
        titlebasics = titlebasics.set_index('tconst')
        # Get list of all 28 genres
        genres = set(",".join(list(map(lambda x: str(x), titlebasics.genres.unique()))).split(','))
        genres.remove('\\N')

        # Get all genres of knownForTitles for all directors, filter out directors without genres
        directors.knownForTitles = directors.knownForTitles.apply(
            lambda x: [get_genres_from_titles(titlebasics, m) for m in x])
        directors.knownForTitles = directors.knownForTitles.apply(lambda x: ",".join(x))
        directors = directors[directors['knownForTitles'].str.len() > 2]
        pickle.dump(directors, open("data-imdb/directors.dat", "wb"))

    # Saves "skills" file and remove quotes chars
    directors.to_csv('data-imdb/IMDB_skill.csv', index=False, header=False, encoding='utf-8')
    fin = open("data-imdb/IMDB_skill.csv", "rt")
    data = fin.read()
    data = data.replace('"', '')
    fin.close()
    fin = open("data-imdb/IMDB_skill.csv", "wt")
    fin.write(data)
    fin.close()

    global skills_imdb
    skills_imdb = ['Crime', 'Horror', 'Biography', 'Animation', 'Drama', 'Comedy', 'Musical', 'News', 'Reality-TV',
                   'Western', 'Game-Show', 'History', 'Short', 'Adult', 'Adventure', 'Romance', 'Sport', 'Fantasy',
                   'Action', 'Film-Noir', 'Family', 'Talk-Show', 'Documentary', 'Mystery', 'Thriller', 'War',
                   'Sci-Fi', 'Music']
    genres = skills_imdb

    # Read skills file and create applicants tuples in set_A
    global set_students_imdb
    with open("data-imdb/IMDB_skill.csv", 'r', encoding='utf-8') as temp_f:
        lines = temp_f.readlines()
        for index, line in enumerate(lines):
            split_line = line.rstrip('\n').split(',')
            if len(split_line) > 1:
                set_students_imdb.add((index, frozenset(split_line[1:])))
    temp_f.close()

    global distances_imdb
    try:
        print('Found IMDB "distances.dat" pre-computed term similarities file.')
        distances_imdb = pickle.load(open("data-imdb/distances.dat", "rb"))
    except (OSError, IOError) as e:
        # Pre compute all shortest distances between skills in the tree
        print('Not found "distances.dat" pre-computed term similarities file. Calculating...')
        for term1 in genres:
            print('\tCalculating similarity for term "{}"'.format(term1))
            for term2 in genres:
                try:
                    term1_synset = wn.synset(term1 + '.n.01')
                    term2_synset = wn.synset(term2 + '.n.01')
                    score = term1_synset.path_similarity(term2_synset)
                except nltk.corpus.reader.wordnet.WordNetError:
                    score = 0
                if term1 in distances_imdb:
                    distances_imdb[term1][term2] = score
                else:
                    distances_imdb[term1] = {term2: score}
        print('Done! Saving IMDB "distances.dat" pre-computed term similarities file.')
        pickle.dump(distances_imdb, open("data-imdb/distances.dat", "wb"))


# Generates a set of amount_projects, with skills amount_s_per_p, based on skill_set
def generate_set_p(amount_projects, amount_s_per_p, skill_set):
    set_p = []
    for i in range(0, amount_projects):
        sample = random.sample(skill_set, amount_s_per_p)
        set_p.append(('Project ' + str(i + 1), frozenset([x[0] for x in sample])))
    return set_p


# Similarity function (using pre computed values)
def similarity(s1, s2):
    return distances_dblp[s1][s2]


# scoreAR(a,r) function
def score_ar(applicant, requirement):
    score = 0
    for skill in applicant[1]:
        score += similarity(skill, requirement)
    return score


# scoreAP(a,p) function
def score_ap(applicant, project):
    score = 0
    for requirement in project[1]:
        score += score_ar(applicant, requirement)
    return score


# scoreTP(t,p) function
def score_tp(team, project):
    score = 0
    for member in team:
        score += score_ap(member, project)
    return score


# Find best teams by naive approach
def naive_teams(applicants, projects, k):
    available_applicants = list(applicants)
    best_teams = {}
    for project in projects:
        possible_teams = itertools.combinations(available_applicants, k)
        team_scores = {}
        for team in possible_teams:
            score_team = score_tp(team, project)
            team_scores[team] = score_team
        best_team_for_project = max(team_scores, key=team_scores.get)
        best_teams[project] = (best_team_for_project, team_scores[best_team_for_project])
        for member in best_team_for_project:
            available_applicants.remove(member)
    return best_teams


# Find best teams for each project individually
def group_heuristic_teams(applicants, projects, k):
    best_teams = {}
    project_applicants_scores = {}
    for project in projects:
        for applicant in applicants:
            score = score_ap(applicant, project)
            if project in project_applicants_scores:
                project_applicants_scores[project][applicant] = score
            else:
                project_applicants_scores[project] = {applicant: score}
    for project in projects:
        for m in range(k):
            project_available = project_applicants_scores[project]
            best_applicant = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_applicant)
            else:
                best_teams[project] = [best_applicant]
            for p_available in project_applicants_scores.values():
                p_available.pop(best_applicant[0], None)
    return best_teams


# Find best teams with single heuristic
def single_heuristic_teams(applicants, projects, k):
    best_teams = {}
    project_applicants_scores = {}
    for project in projects:
        for applicant in applicants:
            score = score_ap(applicant, project)
            if project in project_applicants_scores:
                project_applicants_scores[project][applicant] = score
            else:
                project_applicants_scores[project] = {applicant: score}
    for m in range(k):
        for project in projects:
            project_available = project_applicants_scores[project]
            best_applicant = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_applicant)
            else:
                best_teams[project] = [best_applicant]
            for p_available in project_applicants_scores.values():
                p_available.pop(best_applicant[0], None)
    return best_teams


# Find best teams with pair heuristic
def pair_heuristic_teams(applicants, projects, k):
    best_teams = {}
    project_applicants_scores = {}
    for project in projects:
        for applicant in applicants:
            score = score_ap(applicant, project)
            if project in project_applicants_scores:
                project_applicants_scores[project][applicant] = score
            else:
                project_applicants_scores[project] = {applicant: score}
    for m in range(0, k, 2):
        for project in projects:
            project_available = project_applicants_scores[project]
            best_applicant = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_applicant)
            else:
                best_teams[project] = [best_applicant]
            for p_available in project_applicants_scores.values():
                p_available.pop(best_applicant[0], None)
            if m < k:
                snd_best_applicant = max(project_available.items(), key=operator.itemgetter(1))
                best_teams[project].append(snd_best_applicant)
                for p_available in project_applicants_scores.values():
                    p_available.pop(snd_best_applicant[0], None)
    return best_teams


# Do tests and print results with a given set of projects
def test_with_projects(set_p, k, full_output):
    print('\n---------- Test input: ----------')
    print('\tApplicants: {}\tProjects: {}\tk: {}'.format(len(set_students_dblp), len(set_p), k))
    if full_output:
        for project in set_p:
            print('\t->{}:\trequirements:{}'.format(project[0], list(project[1])))

    start_group_heuristic = time.time()
    res_gh = group_heuristic_teams(set_students_dblp, set_p, k)
    end_group_heuristic = time.time()
    time_group_heuristic = end_group_heuristic - start_group_heuristic
    if log_graphs['time_p']:
        dt_time_p[0][0].append(len(set_p))
        dt_time_p[0][1].append(time_group_heuristic)
    if log_graphs['time_k']:
        dt_time_k[0][0].append(k)
        dt_time_k[0][1].append(time_group_heuristic)

    start_single_heuristic = time.time()
    res_sh = single_heuristic_teams(set_students_dblp, set_p, k)
    end_single_heuristic = time.time()
    time_single_heuristic = end_single_heuristic - start_single_heuristic
    if log_graphs['time_p']:
        dt_time_p[1][0].append(len(set_p))
        dt_time_p[1][1].append(time_single_heuristic)
    if log_graphs['time_k']:
        dt_time_k[1][0].append(k)
        dt_time_k[1][1].append(time_single_heuristic)

    start_pair_heuristic = time.time()
    res_ph = pair_heuristic_teams(set_students_dblp, set_p, k)
    end_pair_heuristic = time.time()
    time_pair_heuristic = end_pair_heuristic - start_pair_heuristic
    if log_graphs['time_p']:
        dt_time_p[2][0].append(len(set_p))
        dt_time_p[2][1].append(time_pair_heuristic)
    if log_graphs['time_k']:
        dt_time_k[2][0].append(k)
        dt_time_k[2][1].append(time_pair_heuristic)

    res_naive = None
    time_naive = None
    if naive:
        start_naive = time.time()
        res_naive = naive_teams(set_students_dblp, set_p, k)
        end_naive = time.time()
        time_naive = end_naive - start_naive
        if log_graphs['time_p']:
            dt_time_p[3][0].append(len(set_p))
            dt_time_p[3][1].append(time_naive)
        if log_graphs['time_k']:
            dt_time_k[3][0].append(k)
            dt_time_k[3][1].append(time_naive)

    print('\nHeuristic results:\tTook: {0:.3f}s'.format(time_group_heuristic))
    gh_teams_scores = []
    for project, team in res_gh.items():
        score_team = score_tp([m[0] for m in team], project)
        gh_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    gh_score_max = max(gh_teams_scores)
    gh_score_min = min(gh_teams_scores)
    gh_score_range = gh_score_max - gh_score_min
    gh_score_mean = mean(gh_teams_scores)
    gh_score_sum = sum(gh_teams_scores)
    gh_score_fd = sum([abs(s - gh_score_mean) for s in gh_teams_scores]) / len(gh_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[0][0].append(len(set_p) * k)
        dt_fairness[0][1].append(gh_score_fd)
    if log_graphs['maxs']:
        dt_maxs[0][0].append(len(set_p) * k)
        dt_maxs[0][1].append(gh_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(gh_score_max,
                                                      gh_score_min,
                                                      gh_score_range,
                                                      gh_score_mean,
                                                      gh_score_sum,
                                                      gh_score_fd))

    print('\nK-rounds heuristic results:\tTook: {0:.3f}s'.format(time_single_heuristic))
    sh_teams_scores = []
    for project, team in res_sh.items():
        score_team = score_tp([m[0] for m in team], project)
        sh_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    sh_score_max = max(sh_teams_scores)
    sh_score_min = min(sh_teams_scores)
    sh_score_range = sh_score_max - sh_score_min
    sh_score_mean = mean(sh_teams_scores)
    sh_score_sum = sum(sh_teams_scores)
    sh_score_fd = sum([abs(s - sh_score_mean) for s in sh_teams_scores]) / len(sh_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[1][0].append(len(set_p) * k)
        dt_fairness[1][1].append(sh_score_fd)
    if log_graphs['maxs']:
        dt_maxs[1][0].append(len(set_p) * k)
        dt_maxs[1][1].append(sh_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(sh_score_max,
                                                      sh_score_min,
                                                      sh_score_range,
                                                      sh_score_mean,
                                                      sh_score_sum,
                                                      sh_score_fd))

    print('\nPair-rounds heuristic results:\tTook: {0:.3f}s'.format(time_pair_heuristic))
    ph_teams_scores = []
    for project, team in res_ph.items():
        score_team = score_tp([m[0] for m in team], project)
        ph_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    ph_score_max = max(ph_teams_scores)
    ph_score_min = min(ph_teams_scores)
    ph_score_range = ph_score_max - ph_score_min
    ph_score_mean = mean(ph_teams_scores)
    ph_score_sum = sum(ph_teams_scores)
    ph_score_fd = sum([abs(s - ph_score_mean) for s in ph_teams_scores]) / len(ph_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[2][0].append(len(set_p) * k)
        dt_fairness[2][1].append(ph_score_fd)
    if log_graphs['maxs']:
        dt_maxs[2][0].append(len(set_p) * k)
        dt_maxs[2][1].append(ph_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(ph_score_max,
                                                      ph_score_min,
                                                      ph_score_range,
                                                      ph_score_mean,
                                                      ph_score_sum,
                                                      ph_score_fd))

    if naive:
        print('\nNaive approach results:\tTook: {0:.3f}s'.format(time_naive))
        naive_teams_scores = []
        for project, team in res_naive.items():
            naive_teams_scores.append(team[1])
            if full_output:
                print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], team[1]))
                for member in team[0]:
                    print('\t\t->Applicant {0!s}:'
                          '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0], score_ap(member, project),
                                                                   list(member[1])))
        naive_score_max = max(naive_teams_scores)
        naive_score_min = min(naive_teams_scores)
        naive_score_range = naive_score_max - naive_score_min
        naive_score_mean = mean(naive_teams_scores)
        naive_score_sum = sum(naive_teams_scores)
        naive_score_fd = sum([abs(s - naive_score_mean) for s in naive_teams_scores]) / len(naive_teams_scores)
        if log_graphs['fairness']:
            dt_fairness[3][0].append(len(set_p) * k)
            dt_fairness[3][1].append(naive_score_fd)
        if log_graphs['maxs']:
            dt_maxs[3][0].append(len(set_p) * k)
            dt_maxs[3][1].append(naive_score_sum)
        print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
              '\tRange:{2:.3f}\tMean:{3:.3f}'
              '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(naive_score_max,
                                                          naive_score_min,
                                                          naive_score_range,
                                                          naive_score_mean,
                                                          naive_score_sum,
                                                          naive_score_fd))


def plot_graphs():
    fig_time, axs_time = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axs_time[0].plot(dt_time_p[0][0], dt_time_p[0][1], label='Heuristic')
    axs_time[0].plot(dt_time_p[1][0], dt_time_p[1][1], label='K-rounds')
    axs_time[0].plot(dt_time_p[2][0], dt_time_p[2][1], label='Pairs-rounds')
    if naive:
        axs_time[0].plot(dt_time_p[3][0], dt_time_p[3][1], label='Naive')
    axs_time[0].set_xlabel("Amount of projects")
    axs_time[0].set_ylabel("Time (s)")
    axs_time[0].legend()

    axs_time[1].plot(dt_time_k[0][0], dt_time_k[0][1], label='Heuristic')
    axs_time[1].plot(dt_time_k[1][0], dt_time_k[1][1], label='K-rounds')
    axs_time[1].plot(dt_time_k[2][0], dt_time_k[2][1], label='Pairs-rounds')
    if naive:
        axs_time[1].plot(dt_time_k[3][0], dt_time_k[3][1], label='Naive')
    axs_time[1].set_xlabel("k")
    axs_time[1].set_ylabel("Time (s)")
    axs_time[1].legend()

    fig_time.tight_layout()
    fig_time.savefig("results_time.png", dpi=250)

    fig_fairness, ax_fairness = plt.subplots(figsize=(10, 12))

    colors = ("red", "green", "blue")
    groups = ("Heuristic", "K-rounds", "Pairs-rounds")
    data = tuple(dt_fairness)
    for data, clr, group in zip(data, colors, groups):
        x = data[0]
        y = data[1]
        ax_fairness.scatter(x, y, alpha=0.8, c=clr, edgecolors='none', s=30, label=group)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_fairness.plot(x, p(x), color=clr, linestyle='dashed', linewidth=1.0)
    if naive:
        data = dt_fairness[3]
        x = data[0]
        y = data[1]
        ax_fairness.scatter(x, y, alpha=0.8, c="yellow", edgecolors='none', s=30, label="Naive")
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_fairness.plot(x, p(x), color="yellow", linestyle='dashed', linewidth=1.0)
    ax_fairness.set_xlabel("choices")
    ax_fairness.set_ylabel("fairness-deviation")
    ax_fairness.legend(loc=2)

    fig_fairness.tight_layout()
    fig_fairness.savefig("results_fairness.png", dpi=250)

    fig_maxs, ax_maxs = plt.subplots(figsize=(10, 12))

    colors = ("red", "green", "blue")
    groups = ("Heuristic", "K-rounds", "Pairs-rounds")
    data = tuple(dt_maxs)
    for data, clr, group in zip(data, colors, groups):
        x = data[0]
        y = data[1]
        ax_maxs.scatter(x, y, alpha=0.8, c=clr, edgecolors='none', s=30, label=group)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_maxs.plot(x, p(x), color=clr, linestyle='dashed', linewidth=1.0)
    if naive:
        data = dt_maxs[3]
        x = data[0]
        y = data[1]
        ax_maxs.scatter(x, y, alpha=0.8, c="yellow", edgecolors='none', s=30, label="Naive")
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_maxs.plot(x, p(x), color="yellow", linestyle='dashed', linewidth=1.0)
    ax_maxs.set_xlabel("choices")
    ax_maxs.set_ylabel("scoreTP sum")
    ax_maxs.legend(loc=2)

    fig_maxs.tight_layout()
    fig_maxs.savefig("results_maxs.png", dpi=250)


def plot_graphs_naive():
    fig_time, ax_time = plt.subplots(figsize=(5, 5))

    objects = ("Heuristic", "K-rounds", "Pairs-rounds", "Naive")
    y_pos = np.arange(len(objects))
    performance = [dt_time_p[0][1][0], dt_time_p[1][1][0], dt_time_p[2][1][0], dt_time_p[3][1][0]]

    ax_time.bar(y_pos, performance, align='center', alpha=0.5)
    ax_time.set_yticks(objects)
    ax_time.set_ylabel('Time (s)')
    ax_time.legend(loc=2)

    fig_time.tight_layout()
    fig_time.savefig("results_time_naive.png", dpi=250)


log_graphs = defaultdict(bool)
dt_time_p = [[[], []], [[], []], [[], []]]
dt_time_k = [[[], []], [[], []], [[], []]]
dt_maxs = [[[], []], [[], []], [[], []]]
dt_fairness = [[[], []], [[], []], [[], []]]
naive = False

if naive:
    dt_time_p.append([[], []])
    dt_time_k.append([[], []])
    dt_maxs.append([[], []])
    dt_fairness.append([[], []])

# Execute algorithms and print results
if __name__ == "__main__":

    prepare_data_dblp()
    prepare_data_imdb()

    log_graphs['fairness'] = True
    log_graphs['maxs'] = True
    log_graphs['time_p'] = True

    if naive:
        try:
            stats = pickle.load(open("stats_naive.dat", "rb"))
            dt_time_p, dt_time_k, dt_fairness, dt_maxs = stats
        except (OSError, IOError) as e:
            set_students_dblp = random.sample(set_students_dblp, 100)
            log_graphs['time_k'] = True
            generated_set_p = generate_set_p(4, 20, skills_dblp)
            test_with_projects(generated_set_p, 4, True)
            pickle.dump((dt_time_p, dt_time_k, dt_fairness, dt_maxs), open("stats_naive.dat", "wb"))
        plot_graphs_naive()
    else:
        try:
            stats = pickle.load(open("stats.dat", "rb"))
            dt_time_p, dt_time_k, dt_fairness, dt_maxs = stats
        except (OSError, IOError) as e:
            for p in range(3, 30, 3):
                generated_set_p = generate_set_p(p, 20, skills_dblp)
                test_with_projects(generated_set_p, 9, False)
            log_graphs['time_p'] = False

            log_graphs['time_k'] = True
            for k in range(3, 30, 3):
                generated_set_p = generate_set_p(11, 20, skills_dblp)
                test_with_projects(generated_set_p, k, False)
            log_graphs['time_k'] = False

            for p in random.sample(range(5, 35), 10):
                for k in random.sample(range(3, 12), 5):
                    generated_set_p = generate_set_p(p, 20, skills_dblp[-200:])
                    test_with_projects(generated_set_p, k, False)

            for p in random.sample(range(5, 35), 10):
                for k in random.sample(range(3, 12), 5):
                    generated_set_p = generate_set_p(p, 20, skills_dblp[:200])
                    test_with_projects(generated_set_p, k, False)

            pickle.dump((dt_time_p, dt_time_k, dt_fairness, dt_maxs), open("stats.dat", "wb"))
        plot_graphs()

    print('\n---------- END ----------')
