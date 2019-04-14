import itertools
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import defaultdict

'''
Number of applicants: 7428
Number of skills: 4480
7428*7427*7426*7425
3041842488715800
avg 0.0001 s per team score calc
304184248871.5800 s = 9639 years

An applicant 'a' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})

Likewise, a project 'p' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})

Skills of the applicant or project are used in the algorithm
 by their numerical identifiers such as 0,1,2,...
'''
set_A = set()
skills_dict = {}


def get_set_a():
    # Read skills file and create applicants tuples in set_A
    applicants_data = []
    with open("DBLP_skill.csv", 'r', encoding='latin1') as temp_f:
        lines = temp_f.readlines()
        for index, line in enumerate(lines):
            split_line = line.rstrip('\n').split(',')
            if len(split_line) > 1:
                applicants_data.append((index, frozenset(split_line[1:])))
    temp_f.close()

    # Gets frequency of terms
    freq_skills = defaultdict(int)
    for applicant in applicants_data:
        applicant_skills = applicant[1]
        for skill in applicant_skills:
            freq_skills[skill] += 1

    # Builds a skill dictionary with ids instead of the terms for faster processing TODO needed?
    for index, term in enumerate(freq_skills.keys()):
        skills_dict[index] = term
    freq_skills_id = {}
    for sid, term in skills_dict.items():
        freq_skills_id[sid] = freq_skills[term]

    # Recreates applicants_data as set_A with ids instead of strings
    for applicant in applicants_data:
        set_A.add((applicant[0], frozenset([list(skills_dict.values()).index(skill) for skill in applicant[1]])))

    # Gets 2k most frequent terms from all the 4480 terms
    most_freq = sorted(freq_skills_id.items(), key=lambda item: item[1], reverse=True)[:2000]


# Creates a tree (graph) with the identifiers of the skills and their relations
G = nx.random_tree(2000)  # TODO

# Pre compute all shortest distances between skills in the tree
distances = dict(nx.all_pairs_shortest_path_length(G))


def get_set_p():
    set_p = frozenset([
        ('Project 1', frozenset([list(skills_dict.values()).index(skill) for skill in
                                 ['learning', 'networks', 'knowledge', 'study', 'graphs', 'analysis']])),
        ('Project 2', frozenset([list(skills_dict.values()).index(skill) for skill in
                                 ['learning', 'fuzzy', 'algorithms', 'networks', 'classification', 'analysis']])),
        ('Project 3', frozenset([list(skills_dict.values()).index(skill) for skill in
                                 ['graphs', 'clustering', 'distributed', 'algorithms', 'analysis']])),
        ('Project 4', frozenset([list(skills_dict.values()).index(skill) for skill in
                                 ['graphs', 'clustering', 'distributed', 'algorithms', 'analysis']])),
    ])
    return set_p


# Similarity function
def similarity(s1, s2):
    if s1 == s2:
        return 1
    elif s1 < 2000 and s2 < 2000 and nx.is_simple_path(G, [s1, s2]):
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
            start = time.time()
            score_team = score_tp(team, project)
            team_scores[team] = score_team
            start_end = time.time()
            print(start_end - start)
        best_team_for_project = max(team_scores, key=team_scores.get)
        best_teams[project] = (best_team_for_project, team_scores[best_team_for_project])
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
            project_applicants_scores = {project: [a for a in lst if a != best_applicant]
                                         for project, lst in project_applicants_scores.items()}
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
            project_applicants_scores = {project: [a for a in lst if a != best_applicant]
                                         for project, lst in project_applicants_scores.items()}
            if m < k:
                project_available.remove((best_applicant[0], best_applicant[1]))  # Is this line needed?
                snd_best_applicant = max(project_available, key=lambda item: item[1])
                best_teams[project].append(snd_best_applicant)
                project_applicants_scores = {project: [a for a in lst if a != snd_best_applicant]
                                             for project, lst in project_applicants_scores.items()}
    return best_teams


def print_results():
    get_set_a()
    set_p = get_set_p()
    '''
    print('Calculating brute force method...')
    start_brute_force = time.time()
    res_bf = brute_force_teams(set_A, set_p, 6)
    end_brute_force = time.time()
    time_brute_force = end_brute_force - start_brute_force
    '''

    print('Input: --------------')
    for project in set_p:
        print('\t"{}",\trequirements:"{}"'.format(project[0], [skills_dict[skill] for skill in project[1]]))
    print('\tAmount of applicants: {}'.format(len(set_A)))
    print('\tAmount of different skills: {}'.format(len(skills_dict)))

    print('\nCalculating single heuristic method...')
    start_single_heuristic = time.time()
    res_sh = single_heuristic_teams(set_A, set_p, 6)
    end_single_heuristic = time.time()
    time_single_heuristic = end_single_heuristic - start_single_heuristic

    print('\nCalculating pair heuristic method...')
    start_pair_heuristic = time.time()
    res_ph = pair_heuristic_teams(set_A, set_p, 6)
    end_pair_heuristic = time.time()
    time_pair_heuristic = end_pair_heuristic - start_pair_heuristic

    print('\n\nResults:')

    '''
    print('Brute force: -------------------')
    print('Execution time:\t{}'.format(time_brute_force))
    for project, team in res_bf:
        print('\tTeam for project "{}",\trequirements:{},\tscoreTP:{}'
              .format(project[0], [skills_dict[skill] for skill in project[1]], team[1]))
        for member in team[0]:
            print('\t\t"{}",\tskills:{},\tscoreAP:{}'
                  .format(member[0], [skills_dict[skill] for skill in member[1]], score_ap(member, project)))
    '''
    print('Single heuristic: --------------')
    print('Execution time:\t{}'.format(time_single_heuristic))
    for project, team in res_sh.items():
        print('\t->"{}":\tscoreTP:{},\trequirements:{}'
              .format(project[0], score_tp([m[0] for m in team], project),
                      [skills_dict[skill] for skill in project[1]]))
        for member in team:
            print('\t\t-{}:\tscoreAP:{}\tskills:{}'
                  .format(member[0][0], member[1], [skills_dict[skill] for skill in member[0][1]]))

    print('Pair heuristic: ----------------')
    print('Execution time:\t{}'.format(time_pair_heuristic))
    for project, team in res_ph.items():
        print('\t->"{}":\tscoreTP:{},\trequirements:{}'
              .format(project[0], score_tp([m[0] for m in team], project),
                      [skills_dict[skill] for skill in project[1]]))
        for member in team:
            print('\t\t-{}:\tscoreAP:{}\tskills:{}'
                  .format(member[0][0], member[1], [skills_dict[skill] for skill in member[0][1]]))


# Execute algorithms and print results
if __name__ == "__main__":
    print_results()
