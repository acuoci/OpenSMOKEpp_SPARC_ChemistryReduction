/*----------------------------------------------------------------------*\
|    ___                   ____  __  __  ___  _  _______                  |
|   / _ \ _ __   ___ _ __ / ___||  \/  |/ _ \| |/ / ____| _     _         |
|  | | | | '_ \ / _ \ '_ \\___ \| |\/| | | | | ' /|  _| _| |_ _| |_       |
|  | |_| | |_) |  __/ | | |___) | |  | | |_| | . \| |__|_   _|_   _|      |
|   \___/| .__/ \___|_| |_|____/|_|  |_|\___/|_|\_\_____||_|   |_|        |
|        |_|                                                              |
|                                                                         |
|   Author: Alberto Cuoci <alberto.cuoci@polimi.it>                       |
|   CRECK Modeling Group <http://creckmodeling.chem.polimi.it>            |
|   Department of Chemistry, Materials and Chemical Engineering           |
|   Politecnico di Milano                                                 |
|   P.zza Leonardo da Vinci 32, 20133 Milano                              |
|                                                                         |
|-------------------------------------------------------------------------|
|                                                                         |
|   This file is part of OpenSMOKE++ Suite.                               |
|                                                                         |
|   Copyright(C) 2014, 2013  Alberto Cuoci                                |
|   Source-code or binary products cannot be resold or distributed        |
|   Non-commercial use only                                               |
|   Cannot modify source-code for any purpose (cannot create              |
|   derivative works)                                                     |
|                                                                         |
\*-----------------------------------------------------------------------*/

#include "dictionary/OpenSMOKE_DictionaryManager.h"
#include "dictionary/OpenSMOKE_DictionaryGrammar.h"
#include "dictionary/OpenSMOKE_DictionaryKeyWord.h"

namespace OpenSMOKE
{
	class Grammar_StaticReduction : public OpenSMOKE::OpenSMOKE_DictionaryGrammar
	{
	protected:

		virtual void DefineRules()
		{
			AddKeyWord(	OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@KineticsFolder",
																OpenSMOKE::SINGLE_PATH,
																"Name of the folder containing the kinetic scheme (XML Version)",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Kinetics",
																OpenSMOKE::SINGLE_PATH,
																"Name of the kinetics CHEMKIN file",
																true));

			AddKeyWord(	OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Thermodynamics",
																OpenSMOKE::SINGLE_PATH,
																"Name of the thermodynamic CHEMKIN file",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Transport",
																OpenSMOKE::SINGLE_PATH,
																"Name of the transport CHEMKIN file",
																false));

			AddKeyWord(	OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@XMLInput",
																OpenSMOKE::SINGLE_PATH,
																"Name of xml file containing the preprocessed data from MATLAB(R)",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Output",
																OpenSMOKE::SINGLE_PATH,
																"Name of folder where to write results",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@DRG",
																OpenSMOKE::SINGLE_BOOL,
																"DRG Analysis",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@DRGEP",
																OpenSMOKE::SINGLE_BOOL,
																"DRG-EP Analysis",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@TestingNeuralNetwork",
																OpenSMOKE::SINGLE_BOOL,
																"Testing the neural network",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Pressure",
																OpenSMOKE::SINGLE_MEASURE,
																"Pressure",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@Epsilon",
																OpenSMOKE::SINGLE_DOUBLE,
																"Threshold value for DRG/DRG-EP Analyses",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@TemperatureThreshold",
																OpenSMOKE::SINGLE_MEASURE,
																"Threshold temperature for DRG/DRG-EP Analyses",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@RetainedThreshold",
																OpenSMOKE::SINGLE_DOUBLE,
																"Threshold value for retaining species",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@KeySpecies",
																OpenSMOKE::VECTOR_STRING, 
																"List of key species", 
																true) );	

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@StrictPolicyThirdBody",
																OpenSMOKE::SINGLE_BOOL,
																"Strict policy on third-body reactions",
																true));

			AddKeyWord(OpenSMOKE::OpenSMOKE_DictionaryKeyWord("@StrictPolicyFallOff",
																OpenSMOKE::SINGLE_BOOL,
																"Strict policy on fall-off reactions",
																true));

			
		}
	};
}

