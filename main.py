import itertools
import time
import operator
import nltk
import pickle
from collections import defaultdict
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
'''
Number of applicants: 7428
Number of different skills: 4480

An applicant 'a' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})

Likewise, a project 'p' is a tuple in the following format:
a = (identifier, {skill1, skill2, ...})
'''
set_A = set()
distances = {}


def get_set_a():
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
        most_freq = sorted(freq_skills.items(), key=lambda item: item[1], reverse=True)[:2000]


def get_set_p():
    set_p = [
        ('Project 1', frozenset(['learning', 'networks', 'knowledge', 'study', 'graphs', 'inference', 'analysis'])),
        ('Project 2', frozenset(['learning', 'fuzzy', 'algorithms', 'networks', 'classification', 'analysis'])),
        ('Project 3', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis'])),
        ('Project 4', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
        ('Project 5', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
        ('Project 6', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
        ('Project 7', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
        ('Project 8', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
        ('Project 9', frozenset(['graphs', 'clustering', 'distributed', 'algorithms', 'analysis', 'data'])),
    ]
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
        print('\t"{}",\trequirements:"{}"'.format(project[0], list(project[1])))
    print('\tAmount of applicants: {}'.format(len(set_A)))

    print('\nCalculating single heuristic method...')
    start_single_heuristic = time.time()
    res_sh = single_heuristic_teams(set_A, set_p, 6)
    end_single_heuristic = time.time()
    time_single_heuristic = end_single_heuristic - start_single_heuristic

    print('Calculating pair heuristic method...')
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
    print('\nSingle heuristic: --------------')
    print('Execution time:\t{}'.format(time_single_heuristic))
    for project, team in res_sh.items():
        print('\t->"{0!s}":\tscoreTP:{1:.3f},\trequirements:{2!s}'
              .format(project[0], score_tp([m[0] for m in team], project), project[1]))
        for member in team:
            print('\t\t->Applicant {0!s}:\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], member[0][1]))

    print('\nPair heuristic: ----------------')
    print('Execution time:\t{}'.format(time_pair_heuristic))
    for project, team in res_ph.items():
        print('\t->"{0!s}":\tscoreTP:{1:.3f},\trequirements:{2!s}'
              .format(project[0], score_tp([m[0] for m in team], project), project[1]))
        for member in team:
            print('\t\t->Applicant {0!s}:\tscoreAP:{1:.3f}\tskills:{2!s}'.format(member[0][0], member[1], member[0][1]))


# Execute algorithms and print results
if __name__ == "__main__":
    print_results()
