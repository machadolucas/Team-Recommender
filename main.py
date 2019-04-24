import itertools
import time
import operator
import nltk
import pickle
import random
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
set_A = set()
distances = {}
most_freq = []


def prepare_data():
    # Read skills file and create applicants tuples in set_A
    global set_A
    with open("DBLP_skill.csv", 'r', encoding='latin1') as temp_f:
        lines = temp_f.readlines()
        for index, line in enumerate(lines):
            split_line = line.rstrip('\n').split(',')
            if len(split_line) > 1:
                set_A.add((index, frozenset(split_line[1:])))
    temp_f.close()

    # Gets frequency of terms
    freq_skills = defaultdict(int)
    for applicant in set_A:
        applicant_skills = applicant[1]
        for skill in applicant_skills:
            freq_skills[skill] += 1

    global distances
    try:
        print('Found "distances.dat" pre-computed term similarities file.')
        distances = pickle.load(open("distances.dat", "rb"))
    except (OSError, IOError) as e:
        # Pre compute all shortest distances between skills in the tree
        print('Not found "distances.dat" pre-computed term similarities file. Calculating...')
        for term1 in freq_skills.keys():
            print('\tCalculating similarity for term "{}"'.format(term1))
            for term2 in freq_skills.keys():
                try:
                    term1_synset = wn.synset(term1 + '.n.01')
                    term2_synset = wn.synset(term2 + '.n.01')
                    score = term1_synset.path_similarity(term2_synset)
                except nltk.corpus.reader.wordnet.WordNetError:
                    score = 0
                if term1 in distances:
                    distances[term1][term2] = score
                else:
                    distances[term1] = {term2: score}
        print('Done! Saving "distances.dat" pre-computed term similarities file.')
        pickle.dump(distances, open("distances.dat", "wb"))

    # Gets 2k most frequent terms from all the 4480 terms
    global most_freq
    most_freq = sorted(freq_skills.items(), key=lambda item: item[1], reverse=True)[:2000]


# Generates a set of amount_projects, with skills amount_s_per_p, based on skill_set
def generate_set_p(amount_projects, amount_s_per_p, skill_set):
    set_p = []
    for i in range(0, amount_projects):
        sample = random.sample(skill_set, amount_s_per_p)
        set_p.append(('Project ' + str(i + 1), frozenset([x[0] for x in sample])))
    return set_p


# Similarity function (using pre computed values)
def similarity(s1, s2):
    return distances[s1][s2]


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
        best_teams[project] = (best_team_for_project, team_scores[best_team_for_project])
        for member in best_team_for_project:
            applicants.remove(member)
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
def test_with_projects(set_p, full_output):
    '''
    start_brute_force = time.time()
    res_bf = brute_force_teams(set_A, set_p, 6)
    end_brute_force = time.time()
    time_brute_force = end_brute_force - start_brute_force
    '''

    print('Test input:')
    print('\tAmount of applicants: {}'.format(len(set_A)))
    for project in set_p:
        print('\t->{}:\trequirements:{}'.format(project[0], list(project[1])))

    start_single_heuristic = time.time()
    res_sh = single_heuristic_teams(set_A, set_p, 10)
    end_single_heuristic = time.time()
    time_single_heuristic = end_single_heuristic - start_single_heuristic

    start_pair_heuristic = time.time()
    res_ph = pair_heuristic_teams(set_A, set_p, 10)
    end_pair_heuristic = time.time()
    time_pair_heuristic = end_pair_heuristic - start_pair_heuristic

    start_group_heuristic = time.time()
    res_gh = group_heuristic_teams(set_A, set_p, 10)
    end_group_heuristic = time.time()
    time_group_heuristic = end_group_heuristic - start_group_heuristic

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

    print('\nSingle heuristic results:\tTook: {0:.3f}s'.format(time_single_heuristic))
    sh_teams_scores = []
    for project, team in res_sh.items():
        score_team = score_tp([m[0] for m in team], project)
        sh_teams_scores.append(score_team)
        print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
        if full_output:
            for member in team:
                print('\t\t->Applicant {0!s}:\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1],
                                                                                     list(member[0][1])))
    sh_score_max = max(sh_teams_scores)
    sh_score_min = min(sh_teams_scores)
    sh_score_range = sh_score_max - sh_score_min
    sh_score_mean = mean(sh_teams_scores)
    sh_score_sum = sum(sh_teams_scores)
    print('Teams scores:\tMax:{0:.3f}\tMin:{1:.3f}\tRange:{2:.3f}\tMean:{3:.3f}\tSum:{4:.3f}'.format(sh_score_max,
                                                                                                     sh_score_min,
                                                                                                     sh_score_range,
                                                                                                     sh_score_mean,
                                                                                                     sh_score_sum))

    print('\nPair heuristic results:\tTook: {0:.3f}s'.format(time_pair_heuristic))
    ph_teams_scores = []
    for project, team in res_ph.items():
        score_team = score_tp([m[0] for m in team], project)
        ph_teams_scores.append(score_team)
        print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
        if full_output:
            for member in team:
                print('\t\t->Applicant {0!s}:\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1],
                                                                                     list(member[0][1])))
    ph_score_max = max(ph_teams_scores)
    ph_score_min = min(ph_teams_scores)
    ph_score_range = ph_score_max - ph_score_min
    ph_score_mean = mean(ph_teams_scores)
    ph_score_sum = sum(ph_teams_scores)
    print('Teams scores:\tMax:{0:.3f}\tMin:{1:.3f}\tRange:{2:.3f}\tMean:{3:.3f}\tSum:{4:.3f}'.format(ph_score_max,
                                                                                                     ph_score_min,
                                                                                                     ph_score_range,
                                                                                                     ph_score_mean,
                                                                                                     ph_score_sum))

    print('\nGroup heuristic results:\tTook: {0:.3f}s'.format(time_group_heuristic))
    gh_teams_scores = []
    for project, team in res_gh.items():
        score_team = score_tp([m[0] for m in team], project)
        gh_teams_scores.append(score_team)
        print('\t->{0!s}:\tscoreTP:{1:.3f}'.format(project[0], score_team))
        if full_output:
            for member in team:
                print('\t\t->Applicant {0!s}:\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1],
                                                                                     list(member[0][1])))
    gh_score_max = max(gh_teams_scores)
    gh_score_min = min(gh_teams_scores)
    gh_score_range = gh_score_max - gh_score_min
    gh_score_mean = mean(gh_teams_scores)
    gh_score_sum = sum(gh_teams_scores)
    print('Teams scores:\tMax:{0:.3f}\tMin:{1:.3f}\tRange:{2:.3f}\tMean:{3:.3f}\tSum:{4:.3f}'.format(gh_score_max,
                                                                                                     gh_score_min,
                                                                                                     gh_score_range,
                                                                                                     gh_score_mean,
                                                                                                     gh_score_sum))


# Execute algorithms and print results
if __name__ == "__main__":
    prepare_data()

    print('\n\n----------------------------------------\n25 skills p/ proj, 2000 skills:')
    set_p_several_skills_avg_frequent = generate_set_p(10, 25, most_freq)
    test_with_projects(set_p_several_skills_avg_frequent, False)

    print('\n\n----------------------------------------\n25 skills p/ proj, 200 most frequent skills:')
    set_p_several_skills_more_frequent = generate_set_p(10, 25, most_freq[:200])
    test_with_projects(set_p_several_skills_more_frequent, False)

    print('\n\n----------------------------------------\n25 skills p/ proj, 200 less frequent skills:')
    set_p_several_skills_less_frequent = generate_set_p(10, 25, most_freq[-200:])
    test_with_projects(set_p_several_skills_less_frequent, False)

    print('\n\n----------------------------------------\n5 skills p/ proj, 2000 skills:')
    set_p_several_skills_avg_frequent = generate_set_p(10, 5, most_freq)
    test_with_projects(set_p_several_skills_avg_frequent, False)

    print('\n\n----------------------------------------\n5 skills p/ proj, 200 most frequent skills:')
    set_p_several_skills_more_frequent = generate_set_p(10, 5, most_freq[:200])
    test_with_projects(set_p_several_skills_more_frequent, False)

    print('\n\n----------------------------------------\n5 skills p/ proj, 200 less frequent skills:')
    set_p_several_skills_less_frequent = generate_set_p(10, 5, most_freq[-200:])
    test_with_projects(set_p_several_skills_less_frequent, False)
