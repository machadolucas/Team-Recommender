import itertools
import matplotlib.pyplot as plt
import networkx as nx
import time

# Creates a tree (graph) with the identifiers of the skills and their relations
G = nx.random_tree(15)  # TODO

# Pre compute all shortest distances between skills in the tree
distances = dict(nx.all_pairs_shortest_path_length(G))

# Plots the graph
plt.plot()
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

'''
An applicant 'a' is a tuple in the following format:
a = (identifier, [skill1, skill2, …])

Likewise, a project 'p' is a tuple in the following format:
a = (identifier, [skill1, skill2, …])

Skills of the applicant or project are used in the algorithm
 by their numerical identifiers such as 0,1,2,…
'''
set_A = [  # TODO
    ('', [0, 1])
]
set_P = [  # TODO
    ('Project 1', [0, 1]),
    ('Project 2', [0, 4]),
    ('Project 3', [2, 3, 1]),
]
skills = {  # TODO
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
    11: "",
    12: "",
    13: "",
    14: "",
    15: "",
    16: "",
    17: "",
    18: "",
    19: "",
    20: "",
}


# Similarity function
def similarity(s1, s2):
    if s1 == s2:
        return 1
    elif nx.is_simple_path(G, [s2, s2]):
        return 1 / distances[s1][s2]
    else:
        return 0


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


# Find best teams by brute force
def brute_force_teams(applicants, projects, k):
    best_teams = {}
    for project in projects:
        possible_teams = itertools.combinations(applicants, k)
        team_scores = {}
        for team in possible_teams:
            score_team = score_tp(team, project)
            team_scores[team] = score_team
        best_team_for_project = max(team_scores, key=team_scores.get)
        best_teams[project] = best_team_for_project
        for member in best_team_for_project:
            applicants.remove(member)
    return best_teams


# Find best teams with single heuristic
def single_heuristic_teams(applicants, projects, k):
    best_teams = {}
    project_applicants_scores = {}
    for project in projects:
        for applicant in applicants:
            score = score_ap(applicant, project)
            if project in project_applicants_scores:
                project_applicants_scores[project].append((applicant, score))
            else:
                project_applicants_scores[project] = [(applicant, score)]
    for m in range(k):
        for project in projects:
            project_available = project_applicants_scores[project]
            best_applicant = max(project_available, key=lambda item: item[1])
            if project in best_teams:
                best_teams[project].append(best_applicant)
            else:
                best_teams[project] = [best_applicant]
            project_applicants_scores = {project: val for project, val in project_applicants_scores.items()
                                         if val[0] != best_applicant}
    return best_teams


# Find best teams with pair heuristic
def pair_heuristic_teams(applicants, projects, k):
    best_teams = {}
    project_applicants_scores = {}
    for project in projects:
        for applicant in applicants:
            score = score_ap(applicant, project)
            if project in project_applicants_scores:
                project_applicants_scores[project].append((applicant, score))
            else:
                project_applicants_scores[project] = [(applicant, score)]
    for m in range(0, k, 2):
        for project in projects:
            project_available = project_applicants_scores[project]
            best_applicant = max(project_available, key=lambda item: item[1])
            if project in best_teams:
                best_teams[project].append(best_applicant)
            else:
                best_teams[project] = [best_applicant]
            project_applicants_scores = {project: val
                                         for project, val in project_applicants_scores.items()
                                         if val[0] != best_applicant}
            if m < k:
                project_available.remove((best_applicant[0], best_applicant[1]))  # Is this line needed?
                snd_best_applicant = max(project_available, key=lambda item: item[1])
                project_applicants_scores = {project: val
                                             for project, val in project_applicants_scores.items()
                                             if val[0] != snd_best_applicant}
    return best_teams


def print_results():
    start_brute_force = time.time()
    res_bf = brute_force_teams(set_A, set_P, 6)
    end_brute_force = time.time()
    time_brute_force = end_brute_force - start_brute_force

    start_single_heuristic = time.time()
    res_sh = single_heuristic_teams(set_A, set_P, 6)
    end_single_heuristic = time.time()
    time_single_heuristic = end_single_heuristic - start_single_heuristic

    start_pair_heuristic = time.time()
    res_ph = pair_heuristic_teams(set_A, set_P, 6)
    end_pair_heuristic = time.time()
    time_pair_heuristic = end_pair_heuristic - start_pair_heuristic

    print('Results:')

    print('Brute force: -------------------')
    print('\tExecution time:\t{}'.format(time_brute_force))

    print('Single heuristic: --------------')
    print('\tExecution time:\t{}'.format(time_single_heuristic))

    print('Pair heuristic: ----------------')
    print('\tExecution time:\t{}'.format(time_pair_heuristic))
