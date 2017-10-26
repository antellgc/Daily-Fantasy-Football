# Roster Optimizer using Integer Linear Programming
# Picks an optimal NFL Fantasy Draft Roster 
# Default Values are for Draft Kings (50k salary Cap, 9 player team)
#
# Developed by: Shawn Stapleton
# Developed Date: 17-Sept-2016
# Latest Revision Date: 25-Oct-2016
#
# v0 - Implemented Integer Linear Programming as base algorithm
# v1 - Added injury report functionality to remove injured players
# v2 - Added Stacking

import pandas as pd
from pulp import * 
import numpy as np

def GenerateRoster(players_df, SalaryCap=50000, NumQB = 1, NumTE = 1, NumWR = 3, NumRB = 2, NumDST = 1, NumFlex = 1 ):

	# Input: 
	#        players_df:  must have the following columns
	#                     Week	Year Name Pos Team h/a Oppt	Points Salary PosID
	# Output:
	#		 roster_df:   The optimal 9 player roster

	# Create variables for all players
	QB_ID  = players_df[players_df['PosID'].str.contains('QB')]['PosID'].values.tolist()
	TE_ID  = players_df[players_df['PosID'].str.contains('TE')]['PosID'].values.tolist()
	RB_ID  = players_df[players_df['PosID'].str.contains('RB')]['PosID'].values.tolist()
	WR_ID  = players_df[players_df['PosID'].str.contains('WR')]['PosID'].values.tolist()
	DST_ID  = players_df[players_df['PosID'].str.contains('DST')]['PosID'].values.tolist()

	POS_ID = QB_ID+TE_ID+RB_ID+WR_ID+DST_ID

	x  = LpVariable.dicts("%s",  POS_ID, 0, 1, LpInteger)
	points  = pd.Series(players_df['Points'].values,index=players_df['PosID']).to_dict()
	salary  = pd.Series(players_df['Salary'].values,index=players_df['PosID']).to_dict()

	dk_solve = LpProblem("ILP", LpMaximize) 
 
	# ****************************************************************
	# Objective 
	# ****************************************************************
	dk_solve += sum( [points[i]*x[i] for i in sorted(POS_ID)] )

	# ****************************************************************
	# Constraints 
	# ****************************************************************

	# Salary Cap at $50k
	dk_solve += sum( [salary[i]*x[i] for i in sorted(POS_ID)] ) <= SalaryCap

	# Only 1 Quaterback
	dk_solve += sum([x[i] for i in sorted(QB_ID)])  == NumQB

	# Between 1 and 2 Tight Ends
	dk_solve += sum([x[i] for i in sorted(TE_ID)])  <= NumTE + NumFlex
	dk_solve += sum([x[i] for i in sorted(TE_ID)])  >= NumTE

	# Between 3 and 4 Wide Receivers
	dk_solve += sum([x[i] for i in sorted(WR_ID)])  <= NumWR + NumFlex
	dk_solve += sum([x[i] for i in sorted(WR_ID)])  >= NumWR
	#dk_solve += sum([x[i] for i in sorted(WR_ID)])  == NumWR

	# Between 2 and 3 Running Backs
	dk_solve += sum([x[i] for i in sorted(RB_ID)])  <= NumRB + NumFlex
	dk_solve += sum([x[i] for i in sorted(RB_ID)])  >= NumRB

	# Only 1 Defence / Special Teams
	dk_solve += sum([x[i] for i in sorted(DST_ID)]) == NumDST

	# Require 9 players
	dk_solve += sum([x[i] for i in sorted(POS_ID)]) == NumQB + NumTE + NumWR + NumRB + NumDST + NumFlex

	# ****************************************************************
	# Solve
	# ****************************************************************
	LpSolverDefault.msg = 1
	GLPK().solve(dk_solve) 

	# ****************************************************************
	# Results
	# ****************************************************************
	print("Solution Status: " + LpStatus[dk_solve.status])

	# Get Selected Player IDs 
	PlayID = [v.name for v in dk_solve.variables() if v.varValue==1]
	roster_df = players_df[players_df['PosID'].isin(PlayID)].reset_index()

	return roster_df, value(dk_solve.objective) , roster_df["Salary"].sum()