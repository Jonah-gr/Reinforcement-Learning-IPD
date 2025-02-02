% This is a LaTeX class that helps writing the bachelor's or master's thesis 
% It is based on the template of the University of Vienna, Faculty of Computer Science
% Licence use is Creative Commons CC BY 4.0
% https://de.overleaf.com/latex/templates/thesis-template-faculty-of-computer-science-university-of-vienna/whyzmtqggxzz

% -- Creating the class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{PHKA_inf}[Template]
% \LoadClass{article} %Caution: the titlepage only fits if this class is used.
%Alternatively the following class should be possible if Din a4 is a requirement.
%caution this has not been tested extensively
%replace the LoadClass{article} it with:
% \LoadClass[a4paper]{article}
\LoadClass[a4paper, fontsize=11pt,listof=totoc,bibliography=totoc,twoside=false]{scrbook}
% \LoadClass[footinclude=false,fontsize=11pt,paper=a4,listof=totoc,bibliography=totoc,BCOR=12mm,DIV=12]{scrbook} % two-sided % original
% \LoadClass[footinclude=false,oneside,fontsize=11pt,paper=a4,listof=totoc,bibliography=totoc,DIV=12]{scrbook} % one-sided


% -- Packages needed in the class
\RequirePackage[a-1b]{pdfx}
\RequirePackage{graphicx} % Required for including pictures
\RequirePackage[pdfa]{hyperref}
\RequirePackage[T1]{fontenc} % Required for accented characters
\RequirePackage{changepage}
\RequirePackage[final]{microtype}
\RequirePackage{xurl}
\RequirePackage{geometry}
\usepackage{layout}
\hypersetup{
	pdfpagelayout   = TwoPageRight   
}





% -- Commands for the title page --------------

\newcommand{\TitleSetup}{Learning to Cooperate: Reinforcement Learning in the Iterated Prisoner's Dilemma}
\newcommand{\TitleTitleSetup}{Dokumentation\par}




% -- Title of the work 
\newcommand{\TitleValue}{Learning to Cooperate: Reinforcement Learning in the Iterated Prisoner's Dilemma\grqq \vspace*{\baselineskip} \vspace*{\baselineskip}}

% -- Name
\newcommand{\WhoValue}{von Jonah Gräfe}

% -- Desired degree
%\newcommand{\DegreeValue}{Bachelor of Science (B.Sc.) }

% -- Year
\newcommand{\YearGerValue}{16.02.2025}

% -- Degree programme code
\newcommand{\ProgrammeCodeValue}{B.Sc. Data Science, AI und Intelligente Systeme}
%\newcommand{\StudyCodeValue}{Informatik}
\newcommand{\MatrikelValue}{909186}
% -- Degree programme name
\newcommand{\ProgrammeNameValue}{Advances in Intelligent Systems}

% -- Supervisors
\newcommand{\SupervisorValue}{Prof. Dr. Dennis Müller\hspace{0.1cm}}
%\newcommand{\CoSupervisorValue}{Akademische(r) Grad(e)  Vorname Zuname \hspace{0.1cm}}
%\newcommand{\CoSupValue}{1} % ture false if there is a cosupervisor or not

% -- Commands to update the values
\newcommand{\Title}[1]{\renewcommand{\TitleValue}{#1\grqq \vspace*{\baselineskip} \vspace*{\baselineskip} }}
\newcommand{\TitleTwo}[1]{\renewcommand{\TitleValue}{#1\grqq \vspace*{\baselineskip} }}
\newcommand{\TitleThree}[1]{\renewcommand{\TitleValue}{#1\grqq}}
\newcommand{\Who}[1]{\renewcommand{\WhoValue}{#1 \vspace*{\baselineskip}}}
\newcommand{\WhoTwo}[1]{\renewcommand{\WhoValue}{#1 }}
\newcommand{\Degree}[1]{\renewcommand{\DegreeValue}{#1 \vspace*{\baselineskip} \vspace*{\baselineskip}} }
\newcommand{\YearGer}[1]{\renewcommand{\YearGerValue}{#1}}
\newcommand{\YearEng}[1]{\renewcommand{\YearEngValue}{#1}}
\newcommand{\ProgrammeCode}[1]{\renewcommand{\ProgrammeCodeValue}{#1}}
\newcommand{\ProgrammeName}[1]{\renewcommand{\ProgrammeNameValue}{#1}}
\newcommand{\Supervisor}[1]{\renewcommand{\SupervisorValue}{#1} \renewcommand{\SupervisorEngValue}{}}
\newcommand{\CoSupervisor}[1]{\renewcommand{\CoSupervisorValue}{#1}\renewcommand{\CoSupervisorEngValue}{}}
\newcommand{\SupervisorTwo}[1]{\renewcommand{\SupervisorEngValue}{#1}}
\newcommand{\CoSupervisorTwo}[1]{\renewcommand{\CoSupervisorEngValue}{#1}}
\newcommand{\Year}[1]{\YearGer{#1} \YearEng{#1}}


\newcommand{\Titlepage}{
    \newgeometry{bottom=0.1cm}
	\begin{titlepage}
		\enlargethispage{5\baselineskip}
		\fontfamily{phv}\selectfont
		\thispagestyle{empty}
		\begin{center}
            \begin{adjustwidth}{-0.5cm}{-2cm}    
				\vspace*{-2cm}
				\hspace*{10.17cm}
				%University logo
				\includegraphics[width=0.35\linewidth]{../poster/logos/HSD.png}\par
				
			    \vspace{4.5cm}
				\centering 	
				
				% Title
				{\scshape{\textbf{\huge \TitleSetup}\par}}
				\vspace{1.5cm}
				%Thesis title
				{\TitleTitleSetup\par}
				% {\LARGE \glqq \TitleValue \par}
				
				\vspace{1.2cm} % If title is multiple lines long change the space here
				
				
				% verfasst von / submitted by  \vspace{0.2cm}\linebreak
				{\fontsize{13}{16}\selectfont \WhoValue}
		
				
				\vspace{2cm} %if name is multiple lines long change the space here
				
				% angestrebter akademischer Grad / in partial fulfilment of the requirements for the degree of \vspace{0.2cm}\linebreak
				% {\fontsize{13}{16}\selectfont \DegreeValue \par}
				\vspace{3.5cm}
				
				%\flushleft
				
				
				\begin{tabular}{lcl}
					Düsseldorf, \YearGerValue \hspace{0.1cm}
					\vspace{2cm}&  &  \\
					Studiengang: & \hspace{1cm} &\ProgrammeCodeValue\\
					Modul: & \hspace{1cm} & \ProgrammeNameValue\\
					% degree programme code as it appears on & & \\
					% the student record sheet:  & & \vspace{0.3cm} \\
					
					% Studienrichtung lt. Studienblatt /  & \hspace{1cm} & \ProgrammeNameValue 
					%  \\
					% degree programme as it appears on & & \\
					% the student record sheet: & & \vspace{0.3cm}\\
					
					Dozent: & \hspace{1cm} & \SupervisorValue \\
					% & \hspace{1cm} & \SupervisorEngValue \\
					% \if\CoSupervisorValue0
					% \else
					% Mitbetreut von / Co-Supervisor: & \hspace{1cm} & \CoSupervisorValue \\
					% & \hspace{1cm} & \CoSupervisorEngValue \\
					% \fi
				\end{tabular}
               
                
			\end{adjustwidth}	

		\end{center}
	\end{titlepage}
	\restoregeometry
}
