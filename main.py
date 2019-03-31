import pandas as pd
import itertools
import sys

set = {}


# Similarity function
def similarity(s1, s2):
    return 0


# scoreAR(a,r) function
def score_ar(applicant, requirement):
    score = 0
    for skill in applicant.skills:
        score += similarity(skill, requirement)

    return score


# scoreAP(a,p) function
def score_ap(applicant, project):
    score = 0
    for requirement in project.requirements:
        score += score_ar(applicant, requirement)

    return score


# scoreTP(t,p) function
def score_tp(team, project):
    score = 0
    for member in team:
        score += score_ap(member, project)

    return score


def bruteForceTeams(applicants, projects, k):
    bestTeams = {}
    for project in projects:
        possibleTeams = itertools.combinations(applicants, k)
        teamScores = {}
        for team in possibleTeams:
            scoreTeam = score_tp(team, project)
            teamScores[team] = scoreTeam
        bestTeamForProject = max(teamScores, key=teamScores.get)
        bestTeams[project] = bestTeamForProject
        for member in bestTeamForProject:
            applicants.remove(member)
    return bestTeams


def heuristicTeams(applicants, projects, k):
    bestTeams = {}
    for m in range(k):
        for project in projects:
            applicantToProjectScores = {}
            for applicant in applicants:
                scoreAP = score_ap(applicant, project)
                applicantToProjectScores[applicant] = scoreAP
            bestApplicant = max(applicantToProjectScores, key=applicantToProjectScores.get)
            if project in bestTeams:
                bestTeams[project].append(bestApplicant)
            else:
                bestTeams[project] = [bestApplicant]
            applicants.remove(bestApplicant)
    return bestTeams
