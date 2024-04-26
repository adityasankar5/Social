// import "./post.scss";
// import FavoriteBorderOutlinedIcon from "@mui/icons-material/FavoriteBorderOutlined";
// import FavoriteOutlinedIcon from "@mui/icons-material/FavoriteOutlined";
// import TextsmsOutlinedIcon from "@mui/icons-material/TextsmsOutlined";
// import ShareOutlinedIcon from "@mui/icons-material/ShareOutlined";
// import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
// import { Link } from "react-router-dom";
// import Comments from "../comments/Comments";
// import { useState } from "react";
// import moment from "moment";
// import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
// import { makeRequest } from "../../axios";
// import { useContext } from "react";
// import { AuthContext } from "../../context/authContext";

// const Post = ({ post }) => {
//   const [commentOpen, setCommentOpen] = useState(false);
//   const [menuOpen, setMenuOpen] = useState(false);

//   const { currentUser } = useContext(AuthContext);

//   const options = {
//     queryKey: ["likes", post.id],
//     queryFn: () =>
//       makeRequest.get("/likes?postId=" + post.id).then((res) => res.data),
//   };

//   const { isLoading, error, data } = useQuery(options);

//   const queryClient = useQueryClient();

//   // Combine like/unlike logic into a single mutation (v5 syntax)
//   const toggleLikeMutation = useMutation((liked) => {
//     return liked
//       ? makeRequest.delete("/likes?postId=" + post.id)
//       : makeRequest.post("/likes", { postId: post.id });
//   }, {
//     onSuccess: () => {
//       // Invalidate and refetch likes query
//       queryClient.invalidateQueries(["likes"]);
//     },
//   });

// const deleteMutation = useMutation(
//   (postId) => makeRequest.delete("/posts/" + postId),
//   {
//     onError: (error, variables, context) => {
//       console.error("Error deleting post:", error);
//       // Handle the error here...
//     },
//     onSuccess: () => {
//       // Invalidate and refetch posts query (if applicable)
//       queryClient.invalidateQueries(["posts"]);
//     },
//   }
// );
//   const handleLike = () => {
//     toggleLikeMutation.mutate(data?.includes(currentUser.id)); // Use optional chaining
//   };

//   const handleDelete = () => {
//     deleteMutation.mutate(post.id);
//   };

//   return (
//     <div className="post">
//       <div className="container">
//         <div className="user">
//           <div className="userInfo">
//             <img src={"/upload/" + post.profilePic} alt="" />
//             <div className="details">
//               <Link
//                 to={`/profile/${post.userId}`}
//                 style={{ textDecoration: "none", color: "inherit" }}
//               >
//                 <span className="name">{post.name}</span>
//               </Link>
//               <span className="date">{moment(post.createdAt).fromNow()}</span>
//             </div>
//           </div>
//           <MoreHorizIcon onClick={() => setMenuOpen(!menuOpen)} />
//           {menuOpen && post.userId === currentUser.id && (
//             <button onClick={handleDelete}>delete</button>
//           )}
//         </div>
//         <div className="content">
//           <p>{post.desc}</p>
//           <img src={"/upload/" + post.img} alt="" />
//         </div>
//         <div className="info">
//           <div className="item">
//             {isLoading ? (
//               "loading"
//             ) : error ? (
//               <span style={{ color: "red" }}>Error fetching likes: {error.message}</span>
//             ) : data?.includes(currentUser.id) ? (
//               <FavoriteOutlinedIcon style={{ color: "red" }} onClick={handleLike} />
//             ) : (I
//               <FavoriteBorderOutlinedIcon onClick={handleLike} />
//             )}
//             {data?.length || 0} Likes
//           </div>
//           <div className="item" onClick={() => setCommentOpen(!commentOpen)}>
//             <TextsmsOutlinedIcon />
//             See Comments
//           </div>
//           <div className="item">
//             <ShareOutlinedIcon />
//             Share
//           </div>
//         </div>
//         {commentOpen && <Comments postId={post.id} />}
//       </div>
//     </div>
//   );
// };

// export default Post;







//ISSUE: The above code is not working properly.Gives an error. Below Code is worksas a template, but isnt what we want.

import "./post.scss";
import FavoriteBorderOutlinedIcon from "@mui/icons-material/FavoriteBorderOutlined";
import FavoriteOutlinedIcon from "@mui/icons-material/FavoriteOutlined";
import TextsmsOutlinedIcon from "@mui/icons-material/TextsmsOutlined";
import ShareOutlinedIcon from "@mui/icons-material/ShareOutlined";
import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
import { Link } from "react-router-dom";
import Comments from "../comments/Comments";
import { useState } from "react";

const Post = ({ post }) => {
  const [commentOpen, setCommentOpen] = useState(false);

  //TEMPORARY
  const liked = false;

  return (
    <div className="post">
      <div className="container">
        <div className="user">
          <div className="userInfo">
            <img src={post.profilePic} alt="" />
            <div className="details">
              <Link
                to={`/profile/${post.userId}`}
                style={{ textDecoration: "none", color: "inherit" }}
              >
                <span className="name">{post.name}</span>
              </Link>
              <span className="date">1 min ago</span>
            </div>
          </div>
          <MoreHorizIcon />
        </div>
        <div className="content">
          <p>{post.desc}</p>
          <img src={"/upload/" + post.img} alt="" />
        </div>
        <div className="info">
          <div className="item">
            {liked ? <FavoriteOutlinedIcon /> : <FavoriteBorderOutlinedIcon />}
            12 Likes
          </div>
          <div className="item" onClick={() => setCommentOpen(!commentOpen)}>
            <TextsmsOutlinedIcon />
            12 Comments
          </div>
          <div className="item">
            <ShareOutlinedIcon />
            Share
          </div>
        </div>
        {commentOpen && <Comments />}
      </div>
    </div>
  );
};
