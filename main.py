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
DBLP Number of students: 7428
DBLP Number of different skills: 4480

IMDB Number of students: 274058
IMDB Number of different skills: 28

An student 'a' is a tuple in the following format:
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
    # Read skills file and create students tuples in set_A
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
    for student in set_students_dblp:
        student_skills = student[1]
        for skill in student_skills:
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
            return False
        else:
            return genres
    except KeyError:
        return False


def prepare_data_imdb():
    global set_students_imdb
    try:
        # Read skills file and create students tuples in set_A
        with open("data-imdb/IMDB_skill.csv", 'r', encoding='utf-8') as temp_f:
            lines = temp_f.readlines()
            for index, line in enumerate(lines):
                split_line = line.rstrip('\n').split(',')
                if len(split_line) > 1:
                    set_students_imdb.add((index, frozenset(split_line[1:])))
        temp_f.close()

    except IOError:
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
                lambda x: [get_genres_from_titles(titlebasics, m) for m in x if get_genres_from_titles(titlebasics, m)])
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
        # Read skills file and create students tuples in set_A
        with open("data-imdb/IMDB_skill.csv", 'r', encoding='utf-8') as temp_f:
            lines = temp_f.readlines()
            for index, line in enumerate(lines):
                split_line = line.rstrip('\n').split(',')
                if len(split_line) > 1:
                    set_students_imdb.add((index, frozenset(split_line[1:])))
        temp_f.close()

    global skills_imdb
    skills_imdb = ['Crime', 'Horror', 'Biography', 'Animation', 'Drama', 'Comedy', 'Musical', 'News', 'Reality-TV',
                   'Western', 'Game-Show', 'History', 'Short', 'Adult', 'Adventure', 'Romance', 'Sport', 'Fantasy',
                   'Action', 'Film-Noir', 'Family', 'Talk-Show', 'Documentary', 'Mystery', 'Thriller', 'War',
                   'Sci-Fi', 'Music']
    skills_imdb = [[x] for x in skills_imdb]
    genres = skills_imdb

    global distances_imdb
    try:
        print('Found IMDB "distances.dat" pre-computed term similarities file.')
        distances_imdb = pickle.load(open("data-imdb/distances.dat", "rb"))
    except (OSError, IOError) as e:
        # Pre compute all shortest distances between skills in the tree
        print('Not found "distances.dat" pre-computed term similarities file. Calculating...')
        for term1 in genres:
            print('\tCalculating similarity for term "{}"'.format(term1[0]))
            for term2 in genres:
                try:
                    term1_synset = wn.synset(term1[0] + '.n.01')
                    term2_synset = wn.synset(term2[0] + '.n.01')
                    score = term1_synset.path_similarity(term2_synset)
                except nltk.corpus.reader.wordnet.WordNetError:
                    score = 0
                if term1[0] in distances_imdb:
                    distances_imdb[term1[0]][term2[0]] = score
                else:
                    distances_imdb[term1[0]] = {term2[0]: score}
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
    if active_dataset == 0:
        return distances_dblp[s1][s2]
    else:
        return distances_imdb[s1][s2]


# scoreAR(a,r) function
def score_ar(student, requirement):
    score = 0
    for skill in student[1]:
        score += similarity(skill, requirement)
    return score


# scoreAP(a,p) function
def score_ap(student, project):
    score = 0
    for requirement in project[1]:
        score += score_ar(student, requirement)
    return score


# scoreTP(t,p) function
def score_tp(team, project):
    score = 0
    for member in team:
        score += score_ap(member, project)
    return score


# Find best teams with naive approach
def naive_teams(students, projects, k):
    available_students = list(students)
    best_teams = {}
    for project in projects:
        possible_teams = itertools.combinations(available_students, k)
        team_scores = {}
        for team in possible_teams:
            score_team = score_tp(team, project)
            team_scores[team] = score_team
        best_team_for_project = max(team_scores, key=team_scores.get)
        best_teams[project] = (best_team_for_project, team_scores[best_team_for_project])
        for member in best_team_for_project:
            available_students.remove(member)
    return best_teams


# Find best teams for each project with heuristic (local optimization)
def local_optimization_heuristic_teams(students, projects, k):
    best_teams = {}
    project_students_scores = {}
    for project in projects:
        for student in students:
            score = score_ap(student, project)
            if project in project_students_scores:
                project_students_scores[project][student] = score
            else:
                project_students_scores[project] = {student: score}
    for project in projects:
        for m in range(k):
            project_available = project_students_scores[project]
            best_student = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_student)
            else:
                best_teams[project] = [best_student]
            for p_available in project_students_scores.values():
                p_available.pop(best_student[0], None)
    return best_teams


# Find best teams with k-rounds (global optimization and k-rounds)
def global_optimization_k_rounds_teams(students, projects, k):
    best_teams = {}
    project_students_scores = {}
    for project in projects:
        for student in students:
            score = score_ap(student, project)
            if project in project_students_scores:
                project_students_scores[project][student] = score
            else:
                project_students_scores[project] = {student: score}
    for m in range(k):
        for project in projects:
            project_available = project_students_scores[project]
            best_student = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_student)
            else:
                best_teams[project] = [best_student]
            for p_available in project_students_scores.values():
                p_available.pop(best_student[0], None)
    return best_teams


# Find best teams with pair-rounds (global optimization and pair-rounds)
def global_optimization_pair_rounds_teams(students, projects, k):
    best_teams = {}
    project_students_scores = {}
    for project in projects:
        for student in students:
            score = score_ap(student, project)
            if project in project_students_scores:
                project_students_scores[project][student] = score
            else:
                project_students_scores[project] = {student: score}
    for m in range(0, k, 2):
        for project in projects:
            project_available = project_students_scores[project]
            best_student = max(project_available.items(), key=operator.itemgetter(1))
            if project in best_teams:
                best_teams[project].append(best_student)
            else:
                best_teams[project] = [best_student]
            for p_available in project_students_scores.values():
                p_available.pop(best_student[0], None)
            if m < k:
                snd_best_student = max(project_available.items(), key=operator.itemgetter(1))
                best_teams[project].append(snd_best_student)
                for p_available in project_students_scores.values():
                    p_available.pop(snd_best_student[0], None)
    return best_teams


# Do tests with a dataset
def test_dataset(dataset_id):
    skills_per_project = 8
    skill_set = skills_dblp
    if dataset_id == 1:
        skill_set = skills_imdb

    global dt_time_p
    global dt_time_k
    global dt_fairness
    global dt_maxs
    if naive:
        # Test group 4
        try:
            stats = pickle.load(open("stats_naive.dat", "rb"))
            dt_time_p, dt_time_k, dt_fairness, dt_maxs = stats
        except (OSError, IOError) as e:
            if dataset_id == 0:
                global set_students_dblp
                set_students_dblp = random.sample(set_students_dblp, 100)
            else:
                global set_students_imdb
                set_students_imdb = random.sample(set_students_imdb, 100)

            log_graphs['time_k'] = True
            generated_set_p = generate_set_p(4, skills_per_project, skill_set)
            test_with_projects(generated_set_p, 4, True)
            pickle.dump((dt_time_p, dt_time_k, dt_fairness, dt_maxs), open("stats_naive.dat", "wb"))

    else:
        try:
            stats = pickle.load(open("stats.dat", "rb"))
            dt_time_p, dt_time_k, dt_fairness, dt_maxs = stats
        except (OSError, IOError) as e:
            # Test group 1
            for p in range(3, 30, 3):
                generated_set_p = generate_set_p(p, skills_per_project, skill_set)
                test_with_projects(generated_set_p, 9, False)
            log_graphs['time_p'] = False

            log_graphs['time_k'] = True
            # Test group 2
            for k in range(3, 30, 3):
                generated_set_p = generate_set_p(11, skills_per_project, skill_set)
                test_with_projects(generated_set_p, k, False)
            log_graphs['time_k'] = False

            # Test group 3
            if dataset_id == 0:
                for p in random.sample(range(5, 35), 10):
                    for k in random.sample(range(3, 12), 5):
                        generated_set_p = generate_set_p(p, skills_per_project, skill_set[-200:])
                        test_with_projects(generated_set_p, k, False)

                for p in random.sample(range(5, 35), 10):
                    for k in random.sample(range(3, 12), 5):
                        generated_set_p = generate_set_p(p, skills_per_project, skill_set[:200])
                        test_with_projects(generated_set_p, k, False)
            else:
                for p in random.sample(range(5, 35), 10):
                    for k in random.sample(range(3, 12), 5):
                        generated_set_p = generate_set_p(p, skills_per_project, skill_set[-200:])
                        test_with_projects(generated_set_p, k, False)

            pickle.dump((dt_time_p, dt_time_k, dt_fairness, dt_maxs), open("stats.dat", "wb"))


# Do tests and print results with a given set of projects
def test_with_projects(set_p, k, full_output):
    set_students = set_students_dblp
    if active_dataset == 1:
        set_students = set_students_imdb

    print('\n---------- Test input: ----------')
    print('\tApplicants: {}\tProjects: {}\tk: {}'.format(len(set_students), len(set_p), k))
    if full_output:
        for project in set_p:
            print('\t->{}:\trequirements:{}'.format(project[0], list(project[1])))

    # Local optimization naive
    res_naive = None
    time_naive = None
    if naive:
        start_naive = time.time()
        res_naive = naive_teams(set_students, set_p, k)
        end_naive = time.time()
        time_naive = end_naive - start_naive
        if log_graphs['time_p']:
            dt_time_p[3][0].append(len(set_p))
            dt_time_p[3][1].append(time_naive)
        if log_graphs['time_k']:
            dt_time_k[3][0].append(k)
            dt_time_k[3][1].append(time_naive)

    # Local optimization heuristic
    start_local_h = time.time()
    res_local_h = local_optimization_heuristic_teams(set_students, set_p, k)
    end_local_h = time.time()
    time_local_h = end_local_h - start_local_h
    if log_graphs['time_p']:
        dt_time_p[0][0].append(len(set_p))
        dt_time_p[0][1].append(time_local_h)
    if log_graphs['time_k']:
        dt_time_k[0][0].append(k)
        dt_time_k[0][1].append(time_local_h)

    # Global optimization k-rounds
    start_global_k = time.time()
    res_global_k = global_optimization_k_rounds_teams(set_students, set_p, k)
    end_global_k = time.time()
    time_global_k = end_global_k - start_global_k
    if log_graphs['time_p']:
        dt_time_p[1][0].append(len(set_p))
        dt_time_p[1][1].append(time_global_k)
    if log_graphs['time_k']:
        dt_time_k[1][0].append(k)
        dt_time_k[1][1].append(time_global_k)

    # Global optimization pair-rounds
    start_global_p = time.time()
    res_global_p = global_optimization_pair_rounds_teams(set_students, set_p, k)
    end_global_p = time.time()
    time_global_p = end_global_p - start_global_p
    if log_graphs['time_p']:
        dt_time_p[2][0].append(len(set_p))
        dt_time_p[2][1].append(time_global_p)
    if log_graphs['time_k']:
        dt_time_k[2][0].append(k)
        dt_time_k[2][1].append(time_global_p)

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

    print('\nLocal optimization heuristic results:\tTook: {0:.3f}s'.format(time_local_h))
    local_h_teams_scores = []
    for project, team in res_local_h.items():
        score_team = score_tp([m[0] for m in team], project)
        local_h_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    local_h_score_max = max(local_h_teams_scores)
    local_h_score_min = min(local_h_teams_scores)
    local_h_score_range = local_h_score_max - local_h_score_min
    local_h_score_mean = mean(local_h_teams_scores)
    local_h_score_sum = sum(local_h_teams_scores)
    local_h_score_fd = sum([abs(s - local_h_score_mean) for s in local_h_teams_scores]) / len(local_h_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[0][0].append(len(set_p) * k)
        dt_fairness[0][1].append(local_h_score_fd)
    if log_graphs['maxs']:
        dt_maxs[0][0].append(len(set_p) * k)
        dt_maxs[0][1].append(local_h_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(local_h_score_max,
                                                      local_h_score_min,
                                                      local_h_score_range,
                                                      local_h_score_mean,
                                                      local_h_score_sum,
                                                      local_h_score_fd))

    print('\nGlobal optimization K-rounds results:\tTook: {0:.3f}s'.format(time_global_k))
    global_k_teams_scores = []
    for project, team in res_global_k.items():
        score_team = score_tp([m[0] for m in team], project)
        global_k_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    global_k_score_max = max(global_k_teams_scores)
    global_k_score_min = min(global_k_teams_scores)
    global_k_score_range = global_k_score_max - global_k_score_min
    global_k_score_mean = mean(global_k_teams_scores)
    global_k_score_sum = sum(global_k_teams_scores)
    global_k_score_fd = sum([abs(s - global_k_score_mean) for s in global_k_teams_scores]) / len(global_k_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[1][0].append(len(set_p) * k)
        dt_fairness[1][1].append(global_k_score_fd)
    if log_graphs['maxs']:
        dt_maxs[1][0].append(len(set_p) * k)
        dt_maxs[1][1].append(global_k_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(global_k_score_max,
                                                      global_k_score_min,
                                                      global_k_score_range,
                                                      global_k_score_mean,
                                                      global_k_score_sum,
                                                      global_k_score_fd))

    print('\nGlobal optimization Pair-rounds results:\tTook: {0:.3f}s'.format(time_global_p))
    global_p_teams_scores = []
    for project, team in res_global_p.items():
        score_team = score_tp([m[0] for m in team], project)
        global_p_teams_scores.append(score_team)
        if full_output:
            print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
            for member in team:
                print('\t\t->Applicant {0!s}:'
                      '\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], list(member[0][1])))

    global_p_score_max = max(global_p_teams_scores)
    global_p_score_min = min(global_p_teams_scores)
    global_p_score_range = global_p_score_max - global_p_score_min
    global_p_score_mean = mean(global_p_teams_scores)
    global_p_score_sum = sum(global_p_teams_scores)
    global_p_score_fd = sum([abs(s - global_p_score_mean) for s in global_p_teams_scores]) / len(global_p_teams_scores)
    if log_graphs['fairness']:
        dt_fairness[2][0].append(len(set_p) * k)
        dt_fairness[2][1].append(global_p_score_fd)
    if log_graphs['maxs']:
        dt_maxs[2][0].append(len(set_p) * k)
        dt_maxs[2][1].append(global_p_score_sum)
    print('Scores:\tMax:{0:.3f}\tMin:{1:.3f}'
          '\tRange:{2:.3f}\tMean:{3:.3f}'
          '\tSum:{4:.3f}\tf-deviation:{5:.3f}'.format(global_p_score_max,
                                                      global_p_score_min,
                                                      global_p_score_range,
                                                      global_p_score_mean,
                                                      global_p_score_sum,
                                                      global_p_score_fd))


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


# Dict with variables to control logging of data to the structures below
log_graphs = defaultdict(bool)
# Data of time consumed by algorithms when varying amount of projects
dt_time_p = [[[], []], [[], []], [[], []]]
# Data of time consumed by algorithms when varying amount of team member
dt_time_k = [[[], []], [[], []], [[], []]]
# Data of Sums of scoreTP values over amount of choices made
dt_maxs = [[[], []], [[], []], [[], []]]
# Data of Fairness-deviation over amount of choices made
dt_fairness = [[[], []], [[], []], [[], []]]

# If true, uses naive method
naive = True
# If 0, uses DBLP. If 1, uses IMDB
active_dataset = 0

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

    active_dataset = 0
    test_dataset(active_dataset)
    # active_dataset = 1
    # test_dataset(active_dataset)

    print('\n---------- END ----------')
'''
    if naive:
        plot_graphs_naive()
    else:
        plot_graphs()
'''